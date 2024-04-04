from DataHandling import DataHandler

class BaseModelInterface:
  def prepare_model(self, learning_rate, batch_size):
     raise NotImplementedError("Must implement prepare_model method")

  def train_model(self, epochs):
      raise NotImplementedError("Must implement train_model method")

  def test_model(self):
      raise NotImplementedError("Must implement test_model method")