import wandb

def download_wandb_logged_images(run_name):
    api = wandb.Api()
    run = api.run(run_name) # "hyoseok/KMUVCL/2j96n8zw"

    print(run.files())
    for file in run.files():
        print(file)
        # if file.name.endswith('.png'):
        #     file.download()
        


from PIL import Image

def images_to_gif(image_fnames, fname):
    image_fnames.sort(key=lambda x: int(x.name.split('_')[-2])) #sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
               save_all=True, duration=DURATION, loop=0)


if __name__ =='__main__' :
    download_wandb_logged_images("hyoseok/KMUVCL/2j96n8zw")


# 이거 쓰려면 애초에 wandb에 로깅할 때 3개의 영역을 나눴어야 할 듯

# 시드 고정이 안되어서 results뽑기가 약간 애매함
