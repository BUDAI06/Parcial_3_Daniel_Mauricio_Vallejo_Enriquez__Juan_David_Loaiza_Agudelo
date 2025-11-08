
#CLASES
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import pandas as pd
import nibabel as nib # Para NIFTI
from datetime import datetime

class GestionDICOM:
    def __init__(self, ruta_carpeta):
        self.ruta_carpeta = ruta_carpeta
        self.slices = self.cargar_slices()
        if not self.slices:
            raise ValueError("No se encontraron archivos DICOM en la carpeta.")
        self.pixel_spacing = self.slices[0].PixelSpacing
        self.slice_thickness = self.slices[0].SliceThickness
        self.volumen_3d = self.construir_volumen_3d()
        self.info_estudio = self.obtener_info_estudio()

    def cargar_slices(self):
        lista_slices = []
        for a, b, c in os.walk(self.ruta_carpeta):
            for nombre_archivo in c:
                ruta_completa = os.path.join(a, nombre_archivo)
                try:
                    dcm = pydicom.dcmread(ruta_completa)
                    if 'PixelData' in dcm:
                        lista_slices.append(dcm)
                except:
                    pass
        # Filtrar por tamaño de pixel_array
        if not lista_slices:
            return []
        ref_shape = lista_slices[0].pixel_array.shape
        slices_filtrados = [s for s in lista_slices if s.pixel_array.shape == ref_shape]
        if len(slices_filtrados) < len(lista_slices):
            print(f"Advertencia: {len(lista_slices) - len(slices_filtrados)} slices fueron descartados por tamaño diferente a {ref_shape}.")
        slices_filtrados.sort(key=lambda x: int(x.InstanceNumber))
        return slices_filtrados

    def construir_volumen_3d(self):
        shape = list(self.slices[0].pixel_array.shape)
        shape.insert(0, len(self.slices))
        volumen = np.zeros(shape, dtype=np.int16)
        for i, s in enumerate(self.slices):
            volumen[i, :, :] = s.pixel_array
        return volumen

    def obtener_info_estudio(self):
        primer_slice = self.slices[0]
        info = {
            'StudyDate': primer_slice.get('StudyDate', 'N/A'),
            'StudyTime': primer_slice.get('StudyTime', 'N/A'),
            'StudyModality': primer_slice.get('Modality', 'N/A'),
            'StudyDescription': primer_slice.get('StudyDescription', 'N/A'),
            'SeriesTime': primer_slice.get('SeriesTime', 'N/A'),
        }
        return info

    def mostrar_vistas_3d(self):
        carpeta_resultados = "Resultados_Pruebas"
        if not os.path.exists(carpeta_resultados):
            os.makedirs(carpeta_resultados)

        v = self.volumen_3d
        idx_trans = v.shape[0] // 2
        idx_coron = v.shape[1] // 2
        idx_sagit = v.shape[2] // 2

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(v[idx_trans, :, :], cmap='gray')
        ax1.set_title(f'Corte Transversal (Slice {idx_trans})')
        ax1.axis('off')

        ax2.imshow(v[:, idx_coron, :], cmap='gray')
        ax2.set_title(f'Corte Coronal (Índice {idx_coron})')
        ax2.axis('off')

        ax3.imshow(v[:, :, idx_sagit], cmap='gray')
        ax3.set_title(f'Corte Sagital (Índice {idx_sagit})')
        ax3.axis('off')

        plt.tight_layout()
        nombre_archivo = os.path.join(carpeta_resultados, "vistas_3_cortes.png")
        fig.savefig(nombre_archivo)
        plt.close(fig) # Limpia la memoria
        print(f"\nImagen de 3 cortes guardada como: {nombre_archivo}")

    def convertir_a_nifti(self, ruta_salida):
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(self.volumen_3d, affine)
        nib.save(nifti_img, ruta_salida)
        print(f"Archivo guardado en {ruta_salida}")


class EstudioImaginologico:
    def __init__(self, gestor_dicom):
        self.gestor = gestor_dicom
        info = self.gestor.info_estudio
        self.study_date = info['StudyDate']
        self.study_time = info['StudyTime']
        self.study_modality = info['StudyModality']
        self.study_description = info['StudyDescription']
        self.series_time = info['SeriesTime']
        self.imagen_3d = self.gestor.volumen_3d
        self.shape = self.imagen_3d.shape
        self.pixel_spacing = self.gestor.pixel_spacing
        self.slice_thickness = self.gestor.slice_thickness
        self.duracion_estudio = self._calcular_duracion(self.study_time, self.series_time)

    def _calcular_duracion(self, t_inicio, t_fin):
        try:
            formato_t = '%H%M%S.%f'
            if '.' not in t_inicio: t_inicio = t_inicio + '.0'
            if '.' not in t_fin: t_fin = t_fin + '.0'
            dt_inicio = datetime.strptime(t_inicio, formato_t)
            dt_fin = datetime.strptime(t_fin, formato_t)
            return str(dt_fin - dt_inicio)
        except Exception as e: return f"Error cálculo: {e}"

    def _normalizar_slice(self, slice_img):
        min_val = np.min(slice_img)
        max_val = np.max(slice_img)
        if max_val - min_val == 0: return np.zeros(slice_img.shape, dtype='uint8')
        img_norm = (slice_img - min_val) / (max_val - min_val) * 255
        return img_norm.astype('uint8')

    def _convertir_a_bgr(self, slice_uint8):
        return cv2.cvtColor(slice_uint8, cv2.COLOR_GRAY2BGR)

    def metodo_zoom(self, slice_index, x1, y1, x2, y2, nombre_archivo_guardado, factor_resize=2.0):
        carpeta_resultados = "Resultados_Pruebas"
        if not os.path.exists(carpeta_resultados):
            os.makedirs(carpeta_resultados)

        slice_original = self.imagen_3d[slice_index, :, :]
        slice_norm = self._normalizar_slice(slice_original)
        recorte = slice_norm[y1:y2, x1:x2]
        slice_bgr = self._convertir_a_bgr(slice_norm)
        cv2.rectangle(slice_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        w_mm = (x2 - x1) * self.pixel_spacing[1]
        h_mm = (y2 - y1) * self.pixel_spacing[0]
        texto = f"{w_mm:.1f} x {h_mm:.1f} mm"
        cv2.putText(slice_bgr, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        nueva_dim = (int(recorte.shape[1] * factor_resize), int(recorte.shape[0] * factor_resize))
        recorte_resized = cv2.resize(recorte, nueva_dim, interpolation=cv2.INTER_CUBIC)

        # Se guarda el recorte
        ruta_recorte = os.path.join(carpeta_resultados, nombre_archivo_guardado)
        cv2.imwrite(ruta_recorte, recorte_resized)
        print(f"\nRecorte guardado en: {ruta_recorte}")

        # Se guarda la imagen comparativa
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(cv2.cvtColor(slice_bgr, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Original (Slice {slice_index}) con Zoom Box')
        ax1.axis('off')
        ax2.imshow(recorte_resized, cmap='gray')
        ax2.set_title(f'Recorte Redimensionado (x{factor_resize})')
        ax2.axis('off')
        plt.tight_layout()

        nombre_comparativa = f"comparativa_{nombre_archivo_guardado}"
        ruta_comparativa = os.path.join(carpeta_resultados, nombre_comparativa)
        fig.savefig(ruta_comparativa)
        plt.close(fig)
        print(f"Imagen comparativa guardada en: {ruta_comparativa}")

    def funcion_segmentacion(self, slice_index, tipo_bin, umbral=127):
        carpeta_resultados = "Resultados_Pruebas"
        if not os.path.exists(carpeta_resultados):
            os.makedirs(carpeta_resultados)

        slice_original = self.imagen_3d[slice_index, :, :]
        slice_norm = self._normalizar_slice(slice_original)
        tipos_opencv = {'0': cv2.THRESH_BINARY, '1': cv2.THRESH_BINARY_INV, '2': cv2.THRESH_TRUNC, '3': cv2.THRESH_TOZERO, '4': cv2.THRESH_TOZERO_INV}
        tipo_elegido = tipos_opencv.get(tipo_bin, cv2.THRESH_BINARY)
        _, img_binarizada = cv2.threshold(slice_norm, umbral, 255, tipo_elegido)
        fig, ax = plt.subplots()
        ax.imshow(img_binarizada, cmap='gray')
        ax.set_title(f'Segmentación (Tipo: {tipo_bin}) - Slice {slice_index}')
        ax.axis('off')

        nombre_archivo = f"segmentacion_tipo_{tipo_bin}.png"
        ruta_segmentacion = os.path.join(carpeta_resultados, nombre_archivo)
        fig.savefig(ruta_segmentacion)
        plt.close(fig)
        print(f"\nImagen de segmentación guardada como: {ruta_segmentacion}")

    def transformacion_morfologica(self, slice_index, k_size, operacion='erosion'):
        carpeta_resultados = "Resultados_Pruebas"
        if not os.path.exists(carpeta_resultados):
            os.makedirs(carpeta_resultados)

        slice_original = self.imagen_3d[slice_index, :, :]
        slice_norm = self._normalizar_slice(slice_original)
        if k_size <= 0: k_size = 3
        kernel = np.ones((k_size, k_size), np.uint8)

        if operacion == 'erosion':
            img_morf = cv2.erode(slice_norm, kernel, iterations=1)
        elif operacion == 'dilatacion':
            img_morf = cv2.dilate(slice_norm, kernel, iterations=1)
        else:
            img_morf = cv2.erode(slice_norm, kernel, iterations=1)

        nombre_archivo = f"morfologia_{operacion}_{k_size}x{k_size}.png"
        ruta_morf = os.path.join(carpeta_resultados, nombre_archivo)
        cv2.imwrite(ruta_morf, img_morf)
        print(f"\nImagen morfológica guardada en: {ruta_morf}")

    def generar_dataframe(self):
        datos = {
            'StudyDate': [self.study_date],
            'StudyTime': [self.study_time],
            'StudyModality': [self.study_modality],
            'StudyDescription': [self.study_description],
            'SeriesTime': [self.series_time],
            'DuracionEstudio': [self.duracion_estudio],
            'ShapeImagen': [str(self.shape)],
        }
        return pd.DataFrame(datos)