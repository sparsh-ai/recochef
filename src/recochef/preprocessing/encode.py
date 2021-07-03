def label_encode(data, cols):
  data = data.copy()
  maps = {}
  for col in cols:
    _unique = data[col].unique()
    _map = {old: new for new, old in enumerate(_unique)}
    _reverse_map = {new: old for new, old in enumerate(_unique)}
    data[col] = data[col].map(_map)
    maps[col+'_TO_IDX'] = _map
    maps['IDX_TO_'+col] = _reverse_map
  return data, maps