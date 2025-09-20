# from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.dataset import TextDataset
# import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


def init_model(model_name, shift=5.0, num_inference_steps=48, device=None):
    print(f"init_model {model_name}")
    model = WanDiffusionWrapper(moe=True). to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    vae = WanVAEWrapper()
    model.model.requires_grad_(False)
    model.low_noise_model.requires_grad_(False)
    model.high_noise_model.requires_grad_(False)
    vae.requires_grad_(False)
    print(f"model.guidance_scale_high: {model.guidance_scale_high}")
    print(f"model.guidance_scale_low: {model.guidance_scale_low}")
    print(f"shift: {shift}")
    print(f"num_inference_steps: {num_inference_steps}")
    print(f"device: {device}")

    scheduler = FlowMatchScheduler(
        shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, vae, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--model_name", type=str, default="Wan2.2-T2V-A14B")
    parser.add_argument("--shift", type=float, default=12.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)

    args = parser.parse_args()

    # launch_distributed_job()
    # launch_distributed_job()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, vae, scheduler, unconditional_dict = init_model(model_name=args.model_name, shift=args.shift, num_inference_steps=args.num_inference_steps, device=device)

    dataset = TextDataset(args.caption_path)

    # if global_rank == 0
    os.makedirs(args.output_folder, exist_ok=True)

    # for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
    for index in tqdm(range(len(dataset))):
        prompt_index = index # * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
        prompt = dataset[prompt_index]
        print(prompt["prompts"])

        conditional_dict = encoder(text_prompts=prompt["prompts"])
        # print(conditional_dict["prompt_embeds"].shape)

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        print(f"full scheduler.timesteps: {scheduler.timesteps}")
        # print(f"selected scheduler.timesteps: {scheduler.timesteps[[0,12,24,36,-1]]}")
        # noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]]
        for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            # print(f"t: {t}")
            # print(f"conditional_dict: {conditional_dict['prompt_embeds'].shape}")
            # print(f"unconditional_dict: {unconditional_dict['prompt_embeds'].shape}")
            _, x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            _, x0_pred_uncond = model(
                latents, unconditional_dict, timestep
            )

            guidance_scale = model.get_guidance_scale()
            print(f"using guidance_scale: {guidance_scale}")
            x0_pred = x0_pred_uncond + guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, 21], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        # noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]]
        # noisy_inputs = noisy_inputs[:, [0, 1, 2, 3, -1]]
        # print(f"noisy_inputs.shape: {noisy_inputs.shape}")
        # print(f"noisy_inputs[0, -1].shape: {noisy_inputs[:, -1].shape}")
        # model.high_noise_model.cpu()
        # model.low_noise_model.cpu()
        # model.cpu()
        # encoder.cpu()
        # vae.cpu()
        # torch.cuda.empty_cache()
        # vae = vae.to(device)
        # target = vae.decode_to_pixel(noisy_inputs[:, -1])
        # target = (target * 0.5 + 0.5).clamp(0, 1)
        # from einops import rearrange
        # video = rearrange(target, 'b t c h w -> b t h w c').cpu()
        # video = (video * 255)
        # print(f"video.shape: {video.shape}")
        # from torchvision.io import write_video
        # write_video(os.path.join('test_output', f"{prompt_index:05d}.mp4"), video[0], fps=16)
        # model.high_noise_model.to(device)
        # model.low_noise_model.to(device)

        stored_data = noisy_inputs

        # this is what SF expects
        torch.save(
            {prompt["prompts"]: stored_data.cpu().detach()
            },
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )

        # torch.save(
        #     {"prompts": prompt["prompts"], 
        #     "ode_latent": stored_data.cpu().detach(),
        #     "text_embedding": conditional_dict["prompt_embeds"],
        #     "negative_text_embedding": unconditional_dict["prompt_embeds"],
        #     # prompt["prompts"]: stored_data_sf.cpu().detach()
        #     },
        #     os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        # )
        # break

    # dist.barrier()


if __name__ == "__main__":
    main()
