"""Copyright (c) 2022 VIKTOR B.V.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from viktor.parametrization import Parametrization, Tab, Section, HiddenField, OptionField, OptionListElement, \
	DownloadButton, NumberField

import numpy as np

gef_option_list = [
	['qc [MPa]', 'puntdruk, qc [MPa]'],
	['fs [MPa]', 'lokale wrijving [MPa]'],
	['u2 [MPa]', 'waterdruk schouder [MPa]'],
	['inclination [degrees]', 'helling [graden]'],
	['Rf [-]', 'wrijvingsgetal [-]']
]
gef_options = [OptionListElement(x, y) for x, y in gef_option_list]

MODEL_OPTIONS = [
	'Default Berkel model',
	'Default Zaandam model'
]


def get_min_depth(params, **kwargs) -> float:
	data = params.section_2.gef_data
	if isinstance(data, dict):
		data = np.array(data['measurement_data']['elevation'], dtype=np.float64)*1e-3
		data = data[~np.isnan(data)]
		data = data - data[0]
		return float(np.round(np.abs(np.amin(data)), decimals=1) - 0.1)  # Round downwards to avoid conflicts
	else:
		return 25


class GEFParametrization(Parametrization):
	"""
	GEFParametrization	
	"""
	section_1 = Section('GEF plot options')
	section_2 = Section('Force plot & Download options')
	section_3 = Section('ML model options')

	section_1.gef_option = OptionField("Plot parameter", options=gef_options, default='qc [MPa]', autoselect_single_option=True)

	section_2.gef_data = HiddenField('gef_data')
	section_2.file_name = HiddenField('file_name')
	section_2.diameter = NumberField("Pile diameter", suffix='m', min=0, default=0.22)
	section_2.btn = DownloadButton('Download Complete .CSV DataFrame', 'download_file')

	section_3.model_option = OptionField("Choose ML model: ", options=MODEL_OPTIONS, default='Default Berkel model', autoselect_single_option=True)
	section_3.drive_depth = NumberField("Driving depth:", min=0.1, max=get_min_depth, num_decimals=1, default=1, suffix='m')
