# diffusion-from-scratch

An educational implementation of denoising diffusion probabilistic models (DDPM) built from first principles in PyTorch, with notebooks that explain the math step by step.

> **Status:** in development. Initial implementation and training runs coming soon.

## Goals

- Implement DDPM from scratch in PyTorch with clear connections to the underlying math
- Train on standard datasets (MNIST, then CIFAR-10) using free GPU compute
- Document each component — forward process, reverse process, U-Net architecture, training loop — with notebooks that build intuition

## Background

This project accompanies my MSc thesis on latent diffusion models for medical CT image synthesis at the Chair of Biomedical Physics, TU Munich. It is intended as a clean reference implementation for learners and as part of my public research portfolio.

## Roadmap

- [ ] Forward (noising) process — derivation and implementation
- [ ] Reverse (denoising) process — training objective
- [ ] U-Net architecture for noise prediction
- [ ] MNIST training run
- [ ] CIFAR-10 training run
- [ ] Sampling and visualization notebook
- [ ] Final write-up

## References

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020) — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- `lucidrains/denoising-diffusion-pytorch` — clean reference implementation
- HuggingFace `diffusers` — production library

## Author

Built by [Hrithik Barua](https://github.com/baruahrithik).
