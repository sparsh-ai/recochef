def label_encode(data, col, maps=None):
  data = data.copy()
  if maps==None:
    maps = {}
    _unique = data[col].unique()
    _map = {old: new for new, old in enumerate(_unique)}
    _reverse_map = {new: old for new, old in enumerate(_unique)}
    data[col] = data[col].map(_map)
    maps[col+'_TO_IDX'] = _map
    maps['IDX_TO_'+col] = _reverse_map
    return data, maps
  else:
    data[col] = data[col].map(maps[col+'_TO_IDX'])
    data = data[data[col].notna()]
    return data