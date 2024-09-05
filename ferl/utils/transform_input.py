import torch

from rclpy.impl import rcutils_logger
logger = rcutils_logger.RcutilsLogger(name="transform_input")


def transform_input(x, trans_dict):
	"""
	transforms x, Tx97D raw space input tensor to the desired input space tensor as specified in trans_dict.
	Input
		- Tx97D torch Tensor from environment
		- trans_dict with all settings

	Output:
		- Tx... torch Tensor according to settings
	"""
	# Transform a torch input x according to sin, cos, noangles, RPY, and lowdim parameters.
	def mat2euler(mat):
		gamma = torch.reshape(torch.atan2(mat[:,2,1], mat[:,2,2]), (-1,1))
		beta = torch.reshape(torch.atan2(-mat[:,2,0], torch.sqrt(mat[:,2,1]**2 + mat[:,2,2]**2)), (-1,1))
		alpha = torch.reshape(torch.atan2(mat[:,1,0], mat[:,0,0]), (-1,1))
		return torch.cat((gamma, beta, alpha), dim=1)

	if len(x.shape) == 1:
		x = torch.unsqueeze(x, axis=0)

	# logger.info(f'transform_input x: {x.shape}')

	# TODO: Figure out these dict ranges
	if trans_dict['6D_laptop']:
		return torch.cat((x[:, 75:78], x[:, 78:81]), dim=1)
	if trans_dict['6D_human']:
		return torch.cat((x[:, 75:78], x[:, 78:81]), dim=1)
	if trans_dict['9D_coffee']:
		return x[:, 51:60]

	x_transform = torch.empty((x.shape[0], 0), requires_grad=True, dtype=torch.float32)

	# angles instead of radians use sin or cos
	if not trans_dict['noangles']:
		if trans_dict['sin']:
			x_transform = torch.cat((x_transform, torch.sin(x[:, :6])), dim=1)
		if trans_dict['cos']:
			x_transform = torch.cat((x_transform, torch.cos(x[:, :6])), dim=1)

	num_joints = 6
	if trans_dict['noxyz']:
		x = x[:, :51]

	# if lowdim, need to remove the joints 1 through 6.
	if trans_dict['lowdim']:
		if not trans_dict['noxyz']:
			x = torch.cat((x[:, :51], x[:, 51+(num_joints-1)*3:]), dim=1)
		x = torch.cat((x[:, :6], x[:, 6+(num_joints-1)*9:]), dim=1)
		num_joints = 1
	elif trans_dict['EErot']:
		x = torch.cat((x[:, :6], x[:, 6+(num_joints-1)*9:]), dim=1)
		num_joints = 1
	# logger.info(f'new x: {x.shape}')

	if trans_dict['norot']:
		x = torch.cat((x[:, :6], x[:, 6+num_joints*9:]), dim=1)
	else:
		if trans_dict['rpy']:
			# Convert to roll pitch yaw representation
			for i in range(6, 6+num_joints*9, 9):
				R = x[:, i:i+9].reshape((x.shape[0],3,3))
				if trans_dict['sin']:
					x_transform = torch.cat((x_transform, torch.sin(mat2euler(R))), dim=1)
				if trans_dict['cos']:
					x_transform = torch.cat((x_transform, torch.cos(mat2euler(R))), dim=1)
				if not trans_dict['sin'] and not trans_dict['cos']:
					x_transform = torch.cat((x_transform, mat2euler(R)), dim=1)

			# Delete columns from x.
			x = torch.cat((x[:, :6], x[:, 6+num_joints*9:]), dim=1)

	if trans_dict['sin'] or trans_dict['cos'] or trans_dict['noangles']:
		x = x[:, 6:]

	# logger.info(f'final x: {x.shape}')
	# logger.info(f'final xt: {x_transform.shape}')
	return torch.cat((x_transform, x), dim=1)


def get_subranges(setting_dict):
	"""
	Get the subranges for a specific input space transformation setting.
	Input
		setting_dict, contains the transformation settings.
	Output:
		List of Lists, the latter containing the start and end indices of the respective subspaces.
	"""
	if setting_dict['6D_laptop'] or setting_dict['6D_human']:
		subranges = [[0, 6]]
		return subranges
	if setting_dict['9D_coffee']:
		subranges = [[0, 9]]
		return subranges

	# angle range
	if setting_dict['noangles']:
		r1 = [0, 0]
	elif setting_dict['sin'] and setting_dict['cos']:
		r1 = [0, 12]
	else:
		r1 = [0, 6]

	# logger.info(f'r1: {r1}')

	# orientation range
	orient_multi = 9
	n_joints = 6
	if setting_dict['norot']:
		r2 = [r1[-1], r1[-1]]
	elif setting_dict['EErot']:
		r2 = [r1[-1], r1[-1]+9]
	else:
		if setting_dict['rpy']:
			orient_multi = 3
		if setting_dict['lowdim']:
			n_joints = 1
		r2delta = n_joints * orient_multi
		r2 = [r1[-1], r1[-1] + r2delta]
	# logger.info(f'r2: {r2}')

	# euclidean range
	if setting_dict['noxyz']:
		r3 = [r2[-1], r2[-1]]
	else:
		r3 = [r2[-1], r2[-1] + n_joints * 3 + 6]
	# logger.info(f'r3: {r3}')

	ret = []
	if setting_dict['noangles'] and setting_dict['norot'] and setting_dict['noxyz']:
		# return []
		ret = []
	elif setting_dict['noangles'] and setting_dict['norot']:
		# return [r3]
		ret = [r3]
	elif setting_dict['noxyz'] and setting_dict['norot']:
		# return [r1]
		ret = [r1]
	elif setting_dict['noangles'] and setting_dict['noxyz']:
		# return [r2]
		ret = [r2]
	elif setting_dict['noangles']:
		# return [r2, r3]
		ret = [r2, r3]
	elif setting_dict['norot']:
		# return [r1, r3]
		ret = [r1, r3]
	elif setting_dict['noxyz']:
		# return [r1, r2]
		ret = [r1, r2]
	else:
		# return [r1, r2, r3]
		ret = [r1, r2, r3]
	# logger.info(f'ret: {ret}')
	return ret