from functools import partial
from typing import Any, Sequence

import jax.numpy as jnp

from flax import linen as nn


ModuleDef = Any


class ConvBlock(nn.Module):
    channels: int
    kernel_size: int
    norm: ModuleDef
    stride: int = 1
    act: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, (self.kernel_size, self.kernel_size), strides=self.stride,
                    padding='SAME', use_bias=False, kernel_init=nn.initializers.kaiming_normal())(x)
        x = self.norm()(x)
        if self.act:
            x = nn.swish(x)
        return x


class ResidualBlock(nn.Module):
    channels: int
    conv_block: ModuleDef

    @nn.compact
    def __call__(self, x):
        channels = self.channels
        conv_block = self.conv_block

        shortcut = x

        residual = conv_block(channels, 3)(x)
        residual = conv_block(channels, 3, act=False)(residual)

        if shortcut.shape != residual.shape:
            shortcut = conv_block(channels, 1, act=False)(shortcut)

        gamma = self.param('gamma', nn.initializers.zeros, 1, jnp.float32)
        out = shortcut + gamma * residual
        out = nn.swish(out)
        return out


class Stage(nn.Module):
    channels: int
    num_blocks: int
    stride: int
    block: ModuleDef

    @nn.compact
    def __call__(self, x):
        stride = self.stride
        if stride > 1:
            x = nn.max_pool(x, (stride, stride), strides=(stride, stride))
        for _ in range(self.num_blocks):
            x = self.block(self.channels)(x)
        return x


class Body(nn.Module):
    channel_list: Sequence[int]
    num_blocks_list: Sequence[int]
    strides: Sequence[int]
    stage: ModuleDef

    @nn.compact
    def __call__(self, x):
        for channels, num_blocks, stride in zip(self.channel_list, self.num_blocks_list, self.strides):
            x = self.stage(channels, num_blocks, stride)(x)
        return x


class Stem(nn.Module):
    channel_list: Sequence[int]
    stride: int
    conv_block: ModuleDef

    @nn.compact
    def __call__(self, x):
        stride = self.stride
        for channels in self.channel_list:
            x = self.conv_block(channels, 3, stride=stride)(x)
            stride = 1
        return x


class Head(nn.Module):
    classes: int
    dropout: ModuleDef

    @nn.compact
    def __call__(self, x):
        x = jnp.mean(x, axis=(1, 2))
        x = self.dropout()(x)
        x = nn.Dense(self.classes)(x)
        return x


class ResNet(nn.Module):
    classes: int
    channel_list: Sequence[int]
    num_blocks_list: Sequence[int]
    strides: Sequence[int]
    head_p_drop: float = 0.

    @nn.compact
    def __call__(self, x, train=True):
        norm = partial(nn.BatchNorm, use_running_average=not train)
        dropout = partial(nn.Dropout, rate=self.head_p_drop, deterministic=not train)
        conv_block = partial(ConvBlock, norm=norm)
        residual_block = partial(ResidualBlock, conv_block=conv_block)
        stage = partial(Stage, block=residual_block)

        x = Stem([32, 32, 64], self.strides[0], conv_block)(x)
        x = Body(self.channel_list, self.num_blocks_list, self.strides[1:], stage)(x)
        x = Head(self.classes, dropout)(x)
        return x
