import argparse
import os
import wandb
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import *
from utils import *
from dataset import MNISTDataModule

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Base config') # do not change

    args = parser.parse_args()
    # 이 부분 concat config + args는 함수로 만들어두던가 하자.
    src_config_dir = os.path.join(os.getcwd(), args.config)
    cfg = load_config(src_config_dir)
    dict_args = vars(args)
    dict_args.update(cfg)
    args = argparse.Namespace(**dict_args)
    # args = args.update(cfg)
    args.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.DEVICE)
    return args 

class GAN(pl.LightningModule):
  def __init__(self,args):
    super().__init__()
    self.G = Generator()
    self.D = Discriminator()
    self.Q = Qrator()
    self.bce_loss = nn.BCELoss()
    self.ce_loss = nn.CrossEntropyLoss()
    
    self.D_labels = torch.ones([args.batch_size, 1]) # Discriminator Label to real
    self.D_fakes = torch.zeros([args.batch_size, 1]) # Discriminator Label to fake

    # self.register_buffer("D_labels", torch.ones([args.batch_size, 1]))
    # self.register_buffer("D_fakes", torch.zeros([args.batch_size, 1]))

  def forward(self, z,c):
    return  self.G(z,c)

  def generator_step(self, x, y):
    # x.shape (batch_size, 784)
    # y.shape (batch_size, 1)

    # noise z와 latent code c를 설정한 파라미터에 맞게 생성
    z, c = sample_noise(x.shape[0], args.n_noise, args.n_c_discrete, args.n_c_continuous, label=y, supervised=True)

    # discrete한 부분들에 대해서 1을 가지는 것을(max) label로 정의
    c_discrete_label = torch.max(c[:, :-2], 1)[1].view(-1, 1) 


    # G를 통과한 fake image가 discriminator를 통과 해 fake인지 아닌지와 feature map을 뽑아둔다.
    z_outputs, features = self.D(self.G(z, c)) # (B,1), (B,10), (B,4)

    # feature map의 경우 Q네트워크를 추가로 통과해 우리가 원하는 c'정보를 추출한다.
    c_discrete_out, cc_mu, cc_var = self.Q(features)

    # 기존의 Generator loss와 동일
    G_loss = self.bce_loss(z_outputs, self.D_labels)

    # 여기가 핵심 Q 네트워크를 통해서 나온 c'정보를 measure해줘야함
    # doiscrete같은 경우, 
    Q_loss_discrete = self.ce_loss(c_discrete_out, c_discrete_label.view(-1))


    Q_loss_continuous = -torch.mean(torch.sum(log_gaussian(c[:, -2:], cc_mu, cc_var), 1)) # N(x | mu,var) -> (B, 2) -> (,1)

    # 상호 정보량을 높이는 방향으로 학습하기 위한 loss
    mutual_info_loss = Q_loss_discrete + Q_loss_continuous*0.1
    GnQ_loss = G_loss + mutual_info_loss

    return GnQ_loss


  def discriminator_step(self, x, y):
    # x.shape (batch_size, 784)
    # y.shape (batch_size, 1)
    x_outputs, _, = self.D(x)
    D_x_loss = self.bce_loss(x_outputs, self.D_labels)

    z, c = sample_noise(x.shape[0], args.n_noise, args.n_c_discrete,args.n_c_continuous, label=y, supervised=True)
    z_outputs, _, = self.D(self.G(z, c))
    D_z_loss = self.bce_loss(z_outputs, self.D_fakes)
    D_loss = D_x_loss + D_z_loss

    return D_loss

  def configure_optimizers(self):
    G_params = list(self.G.parameters()) + list(self.Q.parameters())

    g_optimizer = torch.optim.Adam(G_params, lr=0.0002)
    d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002)
    return [g_optimizer, d_optimizer], []

  def training_step(self, train_batch, batch_idx, optimizer_idx):
    X, Y = train_batch    # X.shape = [batch_size, 1, 28, 28], Y.shape = [batch_size, 1]
    # X = X.squeeze()       # X.shape = [batch_size, 784]    
    
    # train generator
    if optimizer_idx == 0:
      loss = self.generator_step(X, Y)
      self.log('generator_loss', loss)
    
    # train discriminator
    if optimizer_idx == 1:
      loss = self.discriminator_step(X, Y)
      self.log('discriminator_oss', loss)


    return loss
  def on_epoch_end(self):
    # z = self.validation_z.type_as(self.generator.model[0].weight)

    # # log sampled images
    # sample_imgs = self(z)
    # grid = torchvision.utils.make_grid(sample_imgs)
    # self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
    self.eval()
    imgs = get_sample_image(self,args)
            # captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            # Option 1: log images with `WandbLogger.log_image`
    self.logger.log_image(key='sample_images', images=[img for img in imgs], caption=[i for i in range(len(imgs))])
    self.train()


if __name__ =='__main__':
    args = get_parser()
    model = GAN(args)
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = MNISTDataModule(batch_size =  args.batch_size)
    wandb_logger = WandbLogger(project='KMUVCL',group='infogan',log_model='all')
    checkpoint_callback = ModelCheckpoint(dirpath="/content/checkpoints")

    trainer = pl.Trainer(max_epochs=args.max_epoch, gpus=1 if torch.cuda.is_available() else 0, progress_bar_refresh_rate=50,logger = wandb_logger,callbacks=[checkpoint_callback])
    trainer.fit(model, dataset)                                                                                                     