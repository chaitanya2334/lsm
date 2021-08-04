def make_bio_dict(tags, start_idx=0):
    d = dict()
    i = start_idx
    for tag in tags:
        for pre_tag in ['B-', 'I-']:
            d[pre_tag + tag] = i
            i += 1

    d['O'] = i

    return d