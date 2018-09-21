

def get_dataloader():
  data_loader = DataLoader()
  return data_loader


class DataLoader():
  def __init__(self, config):
    self.A_path = config.A_path
    self.B_path = config.B_path

