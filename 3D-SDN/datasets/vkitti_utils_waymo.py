import os

# worldIds = ['0000','0001', '0002', '0004']
worldIds = ['0000','0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009']

sceneIds = ['clone']
# worldSizes = [446, 232, 269, 338, 836]  # 0-446, including 446
worldSizes = [186]  # 0-446, including 446

# category = ['Misc', 'Building', 'Car', 'GuardRail', 'Pole', 'Road', 'Sky', 'Terrain',
#             'TrafficLight', 'TrafficSign', 'Tree', 'Truck', 'Van', 'Vegetation']
category = ['Undefined', 
            'Ego_vehicle',
            'Bus',
            'Other_large_vehicle',
            'Bird',
            'Bicycle',
            'Motorcycle',
            'Motorcyclist',
            'Building',
            'Construction_cone_pole',
            'Cyclist',
            'Car',
            'Dynamic',
            'Ground',
            'Ground_animal',
            'GuardRail',
            'Lane_marker', 
            'Pole',
            'Pedestrian',
            'Pedestrian_object',
            'Road',
            'Road_marker',
            'Sidewalk',
            'Static',
            'Sign',
            'Sky',
            'Terrain',
            'Traffic_light',
            'TrafficSign',
            'Tree',
            'Trailer',
            'Truck',
            'Van',
            'Vegetation']


def get_tables(opt, datadir):
    """
    Get the mapping from (worldId, sceneId, rgb) to the semantic/instance ID.
    The instance ID is uniquely assigned to each car and van in the dataset.
    :param opt: 'segm' or 'inst'
    :param datadir: the dataset root
    :return:
    """
    global_obj_id = 0
    table_inst = {}
    table_segm = {}
    for worldId in worldIds:
        for sceneId in sceneIds:
            with open(os.path.join(datadir, "vkitti_1.3.1_scenegt",
                                   "%s_%s_scenegt_rgb_encoding.txt" % (worldId, sceneId)), 'r') as fin:
                first_line = True
                for line in fin:
                    if first_line:
                        first_line = False
                    else:
                        name, r, g, b = line.split(' ')
                        r, g, b = int(r), int(g), int(b)
                        if name.find(':') == -1:
                            table_segm[(worldId, sceneId, r, g, b)] = category.index(name)
                            table_inst[(worldId, sceneId, r, g, b)] = category.index(name)
                        else:
                            global_obj_id += 1
                            table_segm[(worldId, sceneId, r, g, b)] = category.index(name.split(':')[0])
                            table_inst[(worldId, sceneId, r, g, b)] = 5000 * category.index(
                                name.split(':')[0]) + global_obj_id

    return table_segm if opt == 'segm' else table_inst


def get_lists(opt):
    """
    Get the training/testing split for Virtual KITTI.
    :param opt: 'train' or 'test'
    :return:
    """
    # splitRanges = {'train': [range(0, 356),     range(0, 185),      range(69, 270),     range(0, 270),      range(167, 837)],
    #                'test':  [range(356, 447),   range(185, 233),    range(0, 69),       range(270, 339),    range(0, 167)],
    #                'all':   [range(0, 447),     range(0, 233),      range(0, 270),     range(0, 339),      range(0, 837)]}
    splitRanges = {'train': [iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132]),
                             iter([23,27,29,31,35,73,77,79,81,85,123,127,129,131]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132]),
                             iter([25,29,31,33,37,75,79,81,83,87,125,129,131,133]),
                             iter([23,27,29,31,35,73,77,79,81,85,123,127,129,131]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132])],
                   'test':  [iter([136,174,178,180,182,186]),
                             iter([136,174,178,180,182,186]),
                             iter([136,174,178,180,182,186]),
                             iter([136,174,178,180,182,186]),
                             iter([136,174,178,180,182,186]),
                             iter([135,177,179,181,185]),
                             iter([136,174,178,180,182,186]),
                             iter([137,175,179,181,183,187]),
                             iter([135,177,179,181,185]),
                             iter([136,174,178,180,182,186])],
                   'all':   [iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132,136,174,178,180,182,186]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132,136,174,178,180,182,186]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132,136,174,178,180,182,186]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132,136,174,178,180,182,186]),
                             iter([23,27,29,31,35,73,77,79,81,85,123,127,129,131,135,177,179,181,185]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132,136,174,178,180,182,186]),
                             iter([25,29,31,33,37,75,79,81,83,87,125,129,131,133,137,175,179,181,183,187]),
                             iter([23,27,29,31,35,73,77,79,81,85,123,127,129,131,135,177,179,181,185]),
                             iter([24,28,30,32,36,74,78,80,82,86,124,128,130,132,136,174,178,180,182,186])]}
    _list = []
    for worldId in worldIds:
        for sceneId in sceneIds:
            for imgId in splitRanges[opt][worldIds.index(worldId)]:
                _list += ['%s/%s/%05d.png' % (worldId, sceneId, imgId)]
                # _list += ['%s/%s/%03d.png' % (worldId, sceneId, imgId)]

    return _list

