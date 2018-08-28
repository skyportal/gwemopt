import numpy as np
import healpy as hp

def recombine(groups, sliced_array):
    combined = np.array([])
    for group in groups:
        for i in range(len(sliced_array)):
            if i in group:
                combined

def similar_range(params, map_struct):

    if params['doObservability']:
        observability_struct = map_struct['observability']
        telescope = observability_struct.keys()[0]
        prob = observability_struct[telescope]['prob']

    else:
        prob = map_struct['prob']
    
    nested_map = hp.pixelfunc.reorder(prob, r2n = True)
    regions = params['Nregions']
    region_nsides = int((regions / 12) ** .5)
    res = hp.get_map_size(prob)
    region_size = res / regions
    sliced_array = np.zeros([regions, region_size])

    start = 0
    end = start + region_size

    for region in range(regions):
        sliced_array[region] = nested_map[start:end]
        start += region_size
        end += region_size

    # Theres def a way to do this with numpy but i'm on a train, can't see docs
    sum_hp = lambda arr: np.array([np.sum(row) for row in arr])

    sums = sum_hp(sliced_array)
    sum_neighbours = lambda n: np.sum([sums[p] for p in hp.get_all_neighbours(region_nsides, n, nest=True)])
    nb_sums = np.array([sums[i] + sum_neighbours(i) for i in range(len(sums))])
    region_order = np.argsort(nb_sums)[::-1]
    
    significant_regions = np.array([])
    for idx in region_order:
        if sums[idx] > max(sums) * 0.0001:
            significant_regions = np.append(significant_regions, idx)
    significant_regions = significant_regions.astype(int)

    groups = []

    def group_neighbours(group):
        neighbours = np.array([])
        for i in group:
            neighbours = np.append(neighbours, hp.get_all_neighbours(region_nsides, int(i), nest = True))
        return neighbours

    for idx in significant_regions:
        if len(groups) == 0:
            groups.append([idx])
            continue
        for group in groups:
            if idx in group_neighbours(group):
                group.append(idx)
                break
        else:
            groups.append([idx])
    
    group_maps = []
    for group in groups:
        combined_map = np.array([])
        for i in range(len(sliced_array)):
            if i in group:
                combined_map = np.append(combined_map, sliced_array[i])
            else: combined_map = np.append(combined_map, np.zeros(len(sliced_array[0])))
        group_maps.append(combined_map)

    group_maps = [hp.reorder(hp_map, n2r = True) for hp_map in group_maps]

    return group_maps