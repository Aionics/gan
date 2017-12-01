import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import progressbar
from PIL import Image
import os
import glob

from ganmodels import modelGenerator, modelDiscriminator
from dataset import datasetLoader

CUDA = True
EPOCHS = 5000
BATCH_SIZE = 50

# train_dataset = datasetLoader('keyboards', subfolder='train', preload=False)
train_dataset = datasetLoader('pixels', subfolder='train', preload=False)
train_data_generator = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

data_example = train_dataset.__getitem__(0)
real_example = data_example[0]
noise_example = data_example[1]

print('real_example: ', real_example.shape)
print('noise_example: ', noise_example.shape)

Generator = modelGenerator(noise_example.shape[0], real_example.shape[0])
Discriminator = modelDiscriminator(real_example.shape[0], 1)
if CUDA:
    Generator = Generator.cuda()
    Discriminator = Discriminator.cuda()

GenOptimizer = optim.Adam(Generator.parameters(), lr=0.0006)
DisOptimizer = optim.Adam(Discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()
widgets = [
    progressbar.FormatLabel(''),
    ' [', progressbar.Timer(), '] ',
    progressbar.Percentage(),
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]

files = glob.glob('generated/*')
for f in files:
    os.remove(f)

for epoch in range(EPOCHS):
    gen_losses = []
    dis_losses = []
    dataset_len = len(train_data_generator)

    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=dataset_len, widgets=widgets)
    bar.start()
    for step, data_sample in enumerate(train_data_generator):

        real_image = Variable( data_sample[0] ).float().cuda() if CUDA else Variable( data_sample[0] ).float()
        noise = Variable( data_sample[1] ).float().cuda() if CUDA else Variable( data_sample[1] ).float()

        # < Train Discriminator on real
        Discriminator.zero_grad()

        discriminator_real_decision = Discriminator( real_image ).squeeze()
        discriminator_real_error = criterion( discriminator_real_decision, Variable( torch.ones(real_image.size(0))) if not CUDA else Variable( torch.ones(real_image.size(0))).cuda() )
        discriminator_real_error.backward()
        # </ Train Discriminator on real
        # < Train Discriminator on fake
        fake_image = Generator( noise ) * 255
        discriminator_fake_decision = Discriminator( fake_image.detach() ).squeeze()
        discriminator_fake_error = criterion( discriminator_fake_decision, Variable( torch.zeros(real_image.size(0))) if not CUDA else Variable( torch.zeros(real_image.size(0))).cuda() )
        discriminator_fake_error.backward()
        # </ Train Discriminator on fake
        DisOptimizer.step()

        # < Train Generator
        Generator.zero_grad()

        discriminator_fake_decision = Discriminator( fake_image ).squeeze()
        generator_error = criterion( discriminator_fake_decision, Variable( torch.ones(real_image.size(0))) if not CUDA else Variable( torch.ones(real_image.size(0))).cuda() )
        generator_error.backward()
        # discriminator_fake_error = criterion( discriminator_fake_decision, Variable( torch.zeros(real_image.size(0))) if not CUDA else Variable( torch.zeros(real_image.size(0), 1)).cuda()  )
        # discriminator_fake_error.backward()

        GenOptimizer.step()
        # </ Train Generator

        widgets[0] = progressbar.FormatLabel('Epoch %3.f, step %2.f/%2.f; Dloss: %.4f, Gloss: %.4f; D(x): %.4f, D(G(x)): %.4f' %
            (
            epoch,
            step + 1,
            dataset_len,
            discriminator_fake_error.data.numpy()[0] if not CUDA else discriminator_real_error.cpu().data.numpy()[0],
            generator_error.data.numpy()[0] if not CUDA else generator_error.cpu().data.numpy()[0],
            discriminator_real_decision.data.numpy()[0] if not CUDA else discriminator_real_decision.cpu().data.numpy()[0],
            discriminator_fake_decision.data.numpy()[0] if not CUDA else discriminator_fake_decision.cpu().data.numpy()[0]
        ))
        bar.update(step)
    folder_name = 'generated/'
        # is_dir = os.path.isdir(folder_name)
        # if not is_dir: os.mkdir(folder_name)
    noised = np.random.uniform(low=0.0, high=1.0, size=(1, 100, 1, 1))
    noised = Variable( torch.from_numpy(noised) ).float().cuda() if CUDA else Variable( torch.from_numpy(noised) ).float()
    fake_image = (Generator( noised ) * 255).cpu().data.numpy()[0]
    fake_image = np.transpose(fake_image, (1,2,0))
    fake_image = np.uint8(fake_image)
    im = Image.fromarray( fake_image )
    im.save('%s/epoch%s-%s.png' % (folder_name, epoch, 0))

    if epoch>0 and epoch%200==0:
        folder_name = 'trainedmodels/'


    bar.finish()
