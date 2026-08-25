use std::sync::Arc;

use apxinf_core::{Backend, Error, Result, Tensor};

pub struct CategorySpecificLinear {
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    backend: Arc<dyn Backend>,
}

impl CategorySpecificLinear {
    pub fn new(weights: Vec<Tensor>, biases: Vec<Tensor>, backend: Arc<dyn Backend>) -> Result<Self> {
        if weights.is_empty() || weights.len() != biases.len() {
            return Err(Error::Other("category linear requires matching non-empty weights and biases".into()));
        }
        let shape = weights[0].shape().dims();
        if shape.len() != 2 {
            return Err(Error::Other("category linear weight must be [in,out]".into()));
        }
        let (input_dim, output_dim) = (shape[0], shape[1]);
        if biases[0].shape().dims() != [output_dim] {
            return Err(Error::Other("category linear requires weight [in,out] and bias [out]".into()));
        }
        if weights.iter().any(|weight| weight.shape().dims() != [input_dim, output_dim])
            || biases.iter().any(|bias| bias.shape().dims() != [output_dim]) {
            return Err(Error::Other("category linear parameter shapes differ across embodiments".into()));
        }
        Ok(Self { weights, biases, backend })
    }

    pub fn forward(&self, input: &Tensor, embodiment: usize) -> Result<Tensor> {
        let weight = self.weights.get(embodiment)
            .ok_or_else(|| Error::Other(format!("embodiment {embodiment} is out of range")))?;
        let bias = &self.biases[embodiment];
        let dims = input.shape().dims();
        if dims.len() != 2 || dims[1] != weight.shape().dims()[0] {
            return Err(Error::Other("category linear input has incompatible shape".into()));
        }
        self.backend.add_bias(&self.backend.matmul(input, weight)?, bias)
    }
}

pub struct CategorySpecificMlp {
    first: CategorySpecificLinear,
    second: CategorySpecificLinear,
    backend: Arc<dyn Backend>,
}

impl CategorySpecificMlp {
    pub fn new(first: CategorySpecificLinear, second: CategorySpecificLinear, backend: Arc<dyn Backend>) -> Self {
        Self { first, second, backend }
    }

    pub fn forward(&self, input: &Tensor, embodiment: usize) -> Result<Tensor> {
        let hidden = self.backend.relu(&self.first.forward(input, embodiment)?)?;
        self.second.forward(&hidden, embodiment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::CpuBackend;

    fn tensor(shape: impl Into<apxinf_core::Shape>, values: &[f32]) -> Tensor {
        Tensor::from_f32(shape, values).unwrap()
    }

    #[test]
    fn selects_embodiment_and_applies_two_layer_relu_mlp() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let first = CategorySpecificLinear::new(
            vec![
                tensor((2, 2), &[1.0, 0.0, 0.0, 1.0]),
                tensor((2, 2), &[-1.0, 0.0, 0.0, 2.0]),
            ],
            vec![tensor(vec![2], &[0.0, 0.0]), tensor(vec![2], &[0.5, -0.5])],
            Arc::clone(&backend),
        ).unwrap();
        let second = CategorySpecificLinear::new(
            vec![tensor((2, 1), &[1.0, 1.0]), tensor((2, 1), &[2.0, -1.0])],
            vec![tensor(vec![1], &[0.0]), tensor(vec![1], &[1.0])],
            Arc::clone(&backend),
        ).unwrap();
        let mlp = CategorySpecificMlp::new(first, second, backend);
        let input = tensor((1, 2), &[1.0, 2.0]);
        assert_eq!(mlp.forward(&input, 0).unwrap().as_f32().unwrap(), &[3.0]);
        assert_eq!(mlp.forward(&input, 1).unwrap().as_f32().unwrap(), &[-2.5]);
        assert!(mlp.forward(&input, 2).is_err());
    }
}
