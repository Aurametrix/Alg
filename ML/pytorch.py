 import random
  import torch
  
  class MySillyDNN(torch.nn.Module):
      def __init__(self, input_dim, hidden_dim, output_dim):
          super(MySillyDNN, self).__init__()
          self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
          self.hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
          self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

      def forward(self, x, max_recurrences=3):
          hidden_relu = self.input_layer(x).clamp(min=0)
          for r in range(random.randint(0, max_recurrences)):
              hidden_relu = self.hidden_layer(hidden_relu).clamp(min=0)
          y_pred = self.output_layer(hidden_relu)
          return y_pred
          
 # dogs vs cats model
 data = ImageClassiffierData.from_paths(PATH, tfms=tfms_gtom_model(resnet34, 299))
 learn = ConvLearner.pretrained(resnet34, data, use_fc=True)
 learn.fit(0.01, 2)
