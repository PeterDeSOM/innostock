import pandas as pd

from core_x.env_x import Observations

if __name__ == "__main__":
	_EXPORT_CSV_DIR_ = 'export_csv'

	env_obs = Observations()

	symbols = env_obs.get_symbols()
	symb_count = len(symbols)

	symbols.to_csv('%s/symbols.csv' % _EXPORT_CSV_DIR_, index=False)

	for i in range(symb_count):
		_, _, _, _, isin = env_obs.next_source()
		env_obs.get_current_datasource().to_csv('%s/%s.csv' % (_EXPORT_CSV_DIR_, isin), index=False)
		print('%s is exported.' % isin)

	print('done')