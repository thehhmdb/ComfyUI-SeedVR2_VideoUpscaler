# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Union
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from ....optimization.memory_manager import is_mps_available

class SideResize:
    def __init__(
        self,
        size: int,
        max_size: int = 0,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.size = size
        self.max_size = max_size
        self.downsample_only = downsample_only
        self.interpolation = interpolation
        if is_mps_available():
            self.interpolation = InterpolationMode.BILINEAR

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        """
        Resize image with shortest edge set to size, optionally limiting longest edge.
        
        For large video tensors (batch dimension > 1), processes frames in chunks
        to reduce peak VRAM usage during bicubic interpolation which allocates
        large intermediate buffers proportional to input × output size.
        
        Args:
            image (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image with shortest edge = size,
                                 and no edge exceeding max_size (if max_size > 0).
        """
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        if self.downsample_only and min(width, height) < self.size:
            size = min(width, height)
        else:
            size = self.size

        # Resize to shortest edge (disable antialias only for MPS tensors - not supported)
        antialias = not (isinstance(image, torch.Tensor) and image.device.type == 'mps')
        
        # For large video tensors on GPU, process in chunks to reduce peak VRAM.
        # Bicubic interpolation allocates intermediate buffers proportional to
        # input × output size, which can cause OOM on large batches.
        if isinstance(image, torch.Tensor) and image.is_cuda and image.dim() >= 4:
            batch_size = image.shape[0]
            # Compute expected output dimensions
            aspect = max(width, height) / min(width, height)
            new_short = size
            _out_h = int(height / min(height, width) * new_short)
            _out_w = int(width / min(height, width) * new_short)
            
            # Estimate peak memory needed per frame (input + output + intermediate buffers)
            # Bicubic AA typically needs ~4-8x the output size for intermediate buffers
            input_bytes_per_frame = image.element_size() * image.shape[1] * height * width
            output_bytes_per_frame = image.element_size() * image.shape[1] * _out_h * _out_w
            # Conservative estimate: 6x output for intermediate buffers
            peak_bytes_per_frame = input_bytes_per_frame + output_bytes_per_frame + 6 * output_bytes_per_frame
            peak_gb_per_frame = peak_bytes_per_frame / (1024**3)
            
            # Check if we need chunking: if processing all frames would use > 8GB peak
            if peak_gb_per_frame * batch_size > 8:
                # Calculate safe chunk size
                max_frames = max(1, int(8 / peak_gb_per_frame))
                chunks = []
                for i in range(0, batch_size, max_frames):
                    end = min(i + max_frames, batch_size)
                    chunk = image[i:end]
                    resized_chunk = TVF.resize(chunk, size, self.interpolation, antialias=antialias)
                    
                    # Apply max_size constraint if needed
                    if self.max_size > 0:
                        h, w = resized_chunk.shape[-2:]
                        if max(h, w) > self.max_size:
                            scale = self.max_size / max(h, w)
                            new_h, new_w = round(h * scale), round(w * scale)
                            resized_chunk = TVF.resize(resized_chunk, (new_h, new_w), self.interpolation, antialias=antialias)
                    
                    chunks.append(resized_chunk)
                    del chunk  # Free input chunk immediately
                resized = torch.cat(chunks, dim=0)
                del chunks
                return resized
        
        resized = TVF.resize(image, size, self.interpolation, antialias=antialias)
        
        # Apply max_size constraint if specified
        if self.max_size > 0:
            if isinstance(resized, torch.Tensor):
                h, w = resized.shape[-2:]
            else:
                w, h = resized.size
            
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
                new_h, new_w = round(h * scale), round(w * scale)
                resized = TVF.resize(resized, (new_h, new_w), self.interpolation, antialias=antialias)
        
        return resized
