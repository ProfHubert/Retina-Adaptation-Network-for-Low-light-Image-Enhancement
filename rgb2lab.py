import cv2
import torch
import numpy as np
from scipy import misc
from PIL import Image

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)

def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)

def rgb2xyz(img):
    """
    RGB from 0 to 255
    :param img:
    :return:
    """
    r, g, b = torch.split(img, 1, dim=1)

    x = r * 0.412453 + g * 0.357580 + b * 0.180423
    y = r * 0.212671 + g * 0.715160 + b * 0.072169
    z = r * 0.019334 + g * 0.119193 + b * 0.950227
    return torch.cat([x,y,z], dim=1) 


def xyz2lab(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    # ref_x, ref_y, ref_z = 0.95047, 1., 1.08883
    x = x / ref_x
    y = y / ref_y
    z = z / ref_z

    x = torch.where(x > 0.008856, torch.pow( x , 1/3 ), (7.787 * x) + (16 / 116.))
    y = torch.where(y > 0.008856, torch.pow( y , 1/3 ), (7.787 * y) + (16 / 116.))
    z = torch.where(z > 0.008856, torch.pow( z , 1/3 ), (7.787 * z) + (16 / 116.))

    l = (116. * y) - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    return torch.cat([l/100.0, a/200.0 + 0.5, b/200.0 + 0.5], dim=1)

def lab2xyz(lab):
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    l, a, b = torch.split(lab, 1, dim=1)
    l, a, b = l * 100.0, ( a-0.5) * 200.0, (b-0.5) * 200.0
    y = (l + 16) / 116.
    x = a / 500. + y
    z = y - b / 200.

    y = torch.where(torch.pow( y , 3 ) > 0.008856, torch.pow( y , 3 ), ( y - 16 / 116. ) / 7.787)
    x = torch.where(torch.pow( x , 3 ) > 0.008856, torch.pow( x , 3 ), ( x - 16 / 116. ) / 7.787)
    z = torch.where(torch.pow( z , 3 ) > 0.008856, torch.pow( z , 3 ), ( z - 16 / 116. ) / 7.787)

    x = ref_x * x
    y = ref_y * y
    z = ref_z * z
    return torch.cat([x,y,z],dim=1)


def xyz2rgb(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    return torch.cat([r,g,b], dim=1)


def rgb_to_lab(srgb):

	srgb_pixels = torch.reshape(srgb, [-1, 3]) 

	linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).cuda()
	exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).cuda()
	rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
	
	rgb_to_xyz = torch.tensor([
				#    X        Y          Z
				[0.412453, 0.212671, 0.019334], # R
				[0.357580, 0.715160, 0.119193], # G
				[0.180423, 0.072169, 0.950227], # B
			]).type(torch.FloatTensor).cuda()
	
	xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
	

	# XYZ to Lab
	xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).cuda())

	epsilon = 6.0/29.0

	linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).cuda()

	exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).cuda()

	fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
	# convert to lab
	fxfyfz_to_lab = torch.tensor([
		#  l       a       b
		[  0.0,  500.0,    0.0], # fx
		[116.0, -500.0,  200.0], # fy
		[  0.0,    0.0, -200.0], # fz
	]).type(torch.FloatTensor).cuda()
	lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda()
	#return tf.reshape(lab_pixels, tf.shape(srgb))
    
	cc = torch.reshape(lab_pixels, srgb.shape)
    
	L_chan, a_chan, b_chan =torch.unbind(cc, dim=1)
    
	return torch.cat([L_chan.unsqueeze(1) / 200.0 + 0.5, a_chan.unsqueeze(1) / 200.0 + 0.5, b_chan.unsqueeze(1) / 200.0 + 0.5], dim=1)




def lab_to_rgb(lab):
    
		L_chan, a_chan, b_chan = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
		lab = torch.stack([(L_chan - 0.5) / 1.0 * 200.0, (a_chan - 0.5) / 1.0 * 200.0, (b_chan - 0.5) / 1.0 * 200.0], dim=1)
		lab_pixels = lab.reshape(lab.shape[0], lab.shape[2], lab.shape[3], 3)
		lab_pixels = torch.reshape(lab, [-1, 3])
		# convert to fxfyfz
		lab_to_fxfyfz = torch.tensor([
			#   fx      fy        fz
			[1/116.0, 1/116.0,  1/116.0], # l
			[1/500.0,     0.0,      0.0], # a
			[    0.0,     0.0, -1/200.0], # b
		]).type(torch.FloatTensor).cuda()
		fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda(), lab_to_fxfyfz)

		# convert to xyz
		epsilon = 6.0/29.0
		linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).cuda()
		exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).cuda()


		xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

		# denormalize for D65 white point
		xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).cuda())


		xyz_to_rgb = torch.tensor([
			#     r           g          b
			[ 3.2404542, -0.9692660,  0.0556434], # x
			[-1.5371385,  1.8760108, -0.2040259], # y
			[-0.4985314,  0.0415560,  1.0572252], # z
		]).type(torch.FloatTensor).cuda()

		rgb_pixels =  torch.mm(xyz_pixels, xyz_to_rgb)
		# avoid a slightly negative number messing up the conversion
		#clip
		rgb_pixels[rgb_pixels > 1] = 1
		rgb_pixels[rgb_pixels < 0] = 0

		linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).cuda()
		exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).cuda()
		srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
	
		return torch.reshape(srgb_pixels, lab.shape)

