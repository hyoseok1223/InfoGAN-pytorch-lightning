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
    parser.add_argument('--config', type=str, default='./infogan/config.yaml', help='Base config') # do not change

    args = parser.parse_args()
    src_config_dir = os.path.join(os.getcwd(), args.config)
    cfg = load_config(src_config_dir)

    # concat args & config
    dict_args = vars(args)
    dict_args.update(cfg)
    args = argparse.Namespace(**dict_args)
    args.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args 

class GAN(pl.LightningModule):
  def __init__(self,args=None):
    super().__init__()
    self.G = Generator()
    self.D = Discriminator()
    self.Q = Qrator()
    self.bce_loss = nn.BCELoss()
    self.ce_loss = nn.CrossEntropyLoss()


  def forward(self, z,c):
    return  self.G(z,c)

  def generator_step(self, x, y):
    D_labels = torch.ones([args.batch_size, 1]).to(args.DEVICE) 

    z, c = sample_noise(x.shape[0], args.n_noise, args.n_c_discrete, args.n_c_continuous, device = args.DEVICE,label=y, supervised=True)

    c_discrete_label = torch.max(c[:, :-2], 1)[1].view(-1, 1) 

    z_outputs, features = self.D(self.G(z, c)) # (B,1), (B,10), (B,4)

    c_discrete_out, cc_mu, cc_var = self.Q(features)

    G_loss = self.bce_loss(z_outputs, D_labels) 

    Q_loss_discrete = self.ce_loss(c_discrete_out, c_discrete_label.view(-1))
    Q_loss_continuous = -torch.mean(torch.sum(log_gaussian(c[:, -2:], cc_mu, cc_var), 1)) # N(x | mu,var) -> (B, 2) -> (,1)

    mutual_info_loss = Q_loss_discrete + Q_loss_continuous*0.1
    GnQ_loss = G_loss + mutual_info_loss

    return GnQ_loss


  def discriminator_step(self, x, y):
    D_labels = torch.ones([args.batch_size, 1]).to(args.DEVICE) 
    D_fakes = torch.zeros([args.batch_size, 1]).to(args.DEVICE)
    x_outputs, _, = self.D(x)
    D_x_loss = self.bce_loss(x_outputs, D_labels)

    z, c = sample_noise(x.shape[0], args.n_noise, args.n_c_discrete,args.n_c_continuous, args.DEVICE,label=y, supervised=True)
    z_outputs, _, = self.D(self.G(z, c))
    D_z_loss = self.bce_loss(z_outputs, D_fakes)
    D_loss = D_x_loss + D_z_loss

    return D_loss

  def configure_optimizers(self):
    G_params = list(self.G.parameters()) + list(self.Q.parameters())

    g_optimizer = torch.optim.Adam(G_params, lr=0.0002)
    d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002)
    return [g_optimizer, d_optimizer], []

  def training_step(self, train_batch, batch_idx, optimizer_idx):
    X, Y = train_batch   
    
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

    self.eval()
    imgs = get_sample_image(self,args)

    self.logger.log_image(key='sample_images', images=[img for img in imgs], caption=[i for i in range(len(imgs))])
    self.train()


if __name__ =='__main__':
    args = get_parser()
    model = GAN(args)
    dataset = MNISTDataModule(batch_size =  args.batch_size)
    wandb_logger = WandbLogger(project='KMUVCL',group='infogan',log_model='all')
    checkpoint_callback = ModelCheckpoint(dirpath="./infogan/checkpoints")

    trainer = pl.Trainer(max_epochs=args.max_epoch, gpus=1 if torch.cuda.is_available() else 0, progress_bar_refresh_rate=50,logger = wandb_logger,callbacks=[checkpoint_callback])
    trainer.fit(model, dataset)                      
    # trainer.save_checkpoint("example.ckpt")   -> if you want to just one checkpoint, comment out trainer callback function and run this line                                                                         