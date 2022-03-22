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


from viktor.core import ViktorController, ParamsFromFile, File
from viktor.geo import GEFFile
from viktor.views import Summary, SVGView, SVGResult, SVGAndDataView, SVGAndDataResult, DataGroup, DataItem, \
	MapView, MapResult, MapPoint
from viktor.result import DownloadResult
from viktor.geometry import RDWGSConverter
from viktor import UserException

from ..calculations import ML_processing
from ..calculations import preprocessing

import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

from .parametrization import GEFParametrization, gef_option_list


class GEFController(ViktorController):
	"""GEFFileController"""
	parametrization = GEFParametrization
	summary = Summary()
	viktor_convert_entity_field = True
	label = 'GEF'

	@staticmethod
	def compute_drive_time(speed_predictions: np.array, elevation_array: np.array):
		"""
		Estimation for the drive time (in seconds) by numerically integrating (dx/dt)^(-1) * (dx)
		"""
		times = np.array([0], dtype=np.float64)
		v_mean_prev = 1e-2  # To prevent divide by zero in the first iteration
		for i in range(len(elevation_array)-1):
			v_mean = (speed_predictions[i+1] + speed_predictions[i]) / 2
			dx = np.abs(elevation_array[i] - elevation_array[i+1])
			if v_mean <= 0:
				v_mean = v_mean_prev
			time = times[-1] + dx / v_mean
			v_mean_prev = v_mean
			times = np.append(times, time)
		return times	

	@ParamsFromFile(file_types=['.gef'], max_size=1750000)
	def process_file(self, file: File, **kwargs) -> dict:
		"""
		Load in .gef file -> transformed to dictionary (and exports filename to the HiddenField)
		"""
		file_content = file.getvalue(encoding="ISO-8859-1")
		gef_file = GEFFile(file_content)
		gef_data_dict = gef_file.parse(
			additional_columns=['elevation', 'qc', 'fs', 'u2', 'inclination'],
			return_gef_data_obj=False
		)
		
		filename = str(gef_data_dict['headers']['project_id'])+'_'+str(gef_data_dict['headers']['name'])

		return {
			'section_2': {
				'gef_data': gef_data_dict,
				'file_name': filename
			}
		}

	@SVGView("GEF plot", duration_guess=2)
	def visualize_gef(self, params, **kwargs):
		""" 
		Plot GEF data when using GEF dictionary
		"""
		gef_labels = gef_option_list
		data_dict = params.section_2.gef_data
		plot_option = params.section_1.gef_option

		# Select which label to plot (a bit inefficient but it works)
		label = None
		for x in gef_labels:
			if plot_option in x:
				label = x[1]
		
		# Select measurement data from imported data file
		measurement_data = data_dict["measurement_data"]

		# Transform to Pandas DataFrame
		df = preprocessing.measurement_data_to_processed_DF(measurement_data, diameter=None)
		df = preprocessing.add_composition(df)		
		composition_names = ['dense sand', 'clean sand', 'sand mixtures', 'silt mixtures', 'clays', 'peats']
		composition_names_nl = ['grof zand', 'fijn zand', 'zand mix', 'leem', 'klei', 'veen']
		colors = ['#EB9D4A', '#C0A266', '#7EC4A0', '#479085', '#4B5878', '#B76A3E'] # From Robertson paper		

		y = np.array(df[plot_option], dtype=float)
		elevation = np.array(df['elevation [m]'], dtype=float) # [m]

		# Plot cone tip resistance vs. elevation
		fig, axs = plt.subplots(1, 2, sharey=True, figsize=(7,6))
		axs[0].plot(y, elevation)
		axs[0].set_xlim([0, np.amax(y)])
		axs[0].set_ylim([np.amin(elevation),max((np.amax(elevation), 0))])
		axs[0].set_ylabel('Elevation (NAP) [m]', fontsize=12)
		axs[0].set_xlabel(label, fontsize=12)
		axs[0].grid()

		[axs[1].fill_betweenx(elevation, df[composition_names[k]], 0, color=colors[k], lw=0, label=composition_names_nl[k], step='pre') for k in range(len(composition_names))]
		axs[1].set_xlim([0, 1])
		axs[1].set_xticklabels([])
		axs[1].set_ylim([np.amin(elevation), max((np.amax(elevation), 0))])
		axs[1].legend(fontsize=10, framealpha=1, loc='center left', bbox_to_anchor=(1, 0.5))
		axs[1].grid(axis='y')
		axs[1].set_title('Consolidation (Robertson)', fontsize=12)

		plt.tight_layout()

		# Save fig
		svg_data = StringIO()
		fig.savefig(svg_data, format='svg')
		plt.close()

		return SVGResult(svg_data)

	@SVGView("Force estimation", duration_guess=2)
	def visualize_forces(self, params, **kwargs):
		""" 
		Plot forces, requires pile diameter as NumberField input
		"""
		data_dict = params.section_2.gef_data
		diameter = params.section_2.diameter
		
		# Select measurement data from imported data file
		measurement_data = data_dict["measurement_data"]
		
		# Plot cone tip resistance vs. elevation
		fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(7, 6))

		if diameter is not None:
			df = preprocessing.measurement_data_to_processed_DF(measurement_data, diameter)
			ax1.plot(df['Shaft resist [MN]'], df['elevation [m]'], color='green', lw=2, label = 'Shaft resistance')
			ax1.plot(df['Cone resist [MN]'], df['elevation [m]'], color='orange', lw=1, linestyle='--', label = 'Cone resistance')
			ax1.plot(df['Cone resist MA [MN]'], df['elevation [m]'], color='darkorange', lw=2, label = 'Cone resistance MA')
			ax1.set_xlim([0, max((df['Cone resist [MN]'].max(), df['Shaft resist [MN]'].max()))])
			ax1.set_ylim([df['elevation [m]'].min(), max((df['elevation [m]'].max(), 0))])
			ax1.set_ylabel('Elevation (NAP) [m]', fontsize=12)
			ax1.set_xlabel('Resistance force [MN]', fontsize=12)
			ax1.legend()

			ax2.plot(df['Total resist [MN]'], df['elevation [m]'], color='grey', lw=1, linestyle='--', label='Total resistance')
			ax2.plot(df['Total resist MA [MN]'], df['elevation [m]'], color='black', lw=2, label='Total resistance MA')
			ax2.set_xlim([0, df['Total resist [MN]'].max()])
			ax2.set_ylim([df['elevation [m]'].min(), max((df['elevation [m]'].max(), 0))])
			ax2.set_xlabel('Resistance force [MN]', fontsize=12)
			ax2.legend()
			fig.suptitle('Pile diameter: '+str(diameter)+ ' m')
		ax1.grid()
		ax2.grid()
		plt.tight_layout()

		# Save fig
		svg_data = StringIO()
		fig.savefig(svg_data, format='svg')
		plt.close()
		return SVGResult(svg_data)

	@SVGAndDataView("ML prediction plot", duration_guess=3)
	def visualize_prediction(self, params, **kwargs):
		"""
		Plot predictions of the penetration speed next to the CPT data (qc and fs)
		"""
		data_dict = params.section_2.gef_data
		diameter = params.section_2.diameter
		model_option = params.section_3.model_option

		# Predict penetration speed with ML model
		predictions, elevation_df = ML_processing.ML_prediction(
			data_dict,
			diameter,
			model_option
		)

		# Load CPT data
		df = ML_processing.create_gef_df(
			data_dict,
			diameter
		)

		# Plot ML prediction and CPT data
		fig, axs = plt.subplots(1, 3, sharey=True, figsize=(7, 6))
		axs = axs.flatten()

		fig.suptitle('Pile diameter: '+str(diameter)+r' m')
		axs[0].plot(predictions, elevation_df, color='black', lw=2)
		axs[0].set_xlim([0, 1.5 * max(predictions)])
		axs[0].set_ylabel('Elevation (from mudline) [m]', fontsize=12)
		axs[0].set_xlabel('Penetration speed [m/s]', fontsize=12)
		axs[0].grid()

		axs[1].plot(df['fs [MPa]'], elevation_df, color='green', lw=2)
		axs[1].set_xlim([0, 1.5 * max(df['fs [MPa]'])])
		axs[1].set_xlabel('Shaft pressure [MPa]', fontsize=12)
		axs[1].grid()

		axs[2].plot(df['qc [MPa]'], elevation_df, color='orange', lw=2)
		axs[2].set_xlim([0, 1.5 * max(df['qc [MPa]'])])
		axs[2].set_xlabel('Cone pressure [MPa]', fontsize=12)
		axs[2].grid()

		plt.tight_layout()

		# Save fig
		svg_data = StringIO()
		fig.savefig(svg_data, format='svg')
		plt.close()

		# Compute drive time
		times = self.compute_drive_time(np.array(predictions), np.array(elevation_df))
		drive_depth = params.section_3.drive_depth
		try:
			drive_time = times[np.where(np.round(np.array(elevation_df), decimals=1) == -drive_depth)[0][0]]
		except:
			print('Elevation array is:', np.array(elevation_df))
			raise UserException('No valid drive depth selected.')	

		output_time = DataGroup(
					DataItem('Excpected drive time (for continuous driving):', drive_time, suffix='seconds', number_of_decimals=1),
					DataItem('Driving depth:', drive_depth, suffix='m', number_of_decimals=1)
				)
		return SVGAndDataResult(svg_data, output_time)

	@MapView('CPT location', duration_guess=3)
	def visualize_location(self, params, **kwargs):
		data_dict = params.section_2.gef_data
		gef_coordinates_rd = data_dict.headers.x_y_coordinates if 'x_y_coordinates' in data_dict.headers else None
		
		if gef_coordinates_rd is None:
			raise UserException('GEF File does not contain x-y coordinates!')

		gef_coordinates_wgs = RDWGSConverter.from_rd_to_wgs((gef_coordinates_rd[0], gef_coordinates_rd[1]))
		return MapResult([MapPoint(gef_coordinates_wgs[0], gef_coordinates_wgs[1])])

	def download_file(self, params, **kwargs):
		"""
		Download file button.
		"""
		data_dict = params.section_2.gef_data
		filename = params.section_2.file_name
		diameter = params.section_2.diameter
		
		if diameter is None:
			measurement_data = data_dict['measurement_data']
			# Dictionary to DataFrame
			df = preprocessing.measurement_data_to_processed_DF(measurement_data, diameter=diameter)
			
			file_string = df.to_csv()
			return DownloadResult(file_string, str(filename)+'.csv')

		measurement_data = data_dict['measurement_data']
		df = preprocessing.measurement_data_to_processed_DF(measurement_data, diameter=diameter)

		file_string = df.to_csv()
		return DownloadResult(file_string, str(filename)+'_processed.csv')
