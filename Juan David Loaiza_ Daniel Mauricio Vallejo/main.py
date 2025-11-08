
#MENÚ
import sys
import pandas as pd
import os

from clases import GestionDICOM, EstudioImaginologico

lista_estudios = []

def mostrar_menu_principal():
    print("\n--- MENU PRINCIPAL ---")
    print("1. Cargar carpeta DICOM y reconstruccion 3D" )
    print("2. Funciones de Estudio (Zoom, Segmentacion, Morfologia)")
    print("3. Convertir estudio a NIFTI")
    print("4. Guardar información de todos los estudios cargados a CSV")
    print("5. Salir")

def menu_funciones_estudio(estudio):
    if not isinstance(estudio, EstudioImaginologico):
        print("No hay un objeto de estudio válido.")
        return

    print(f"\n--- Operando sobre estudio (Shape: {estudio.shape}) ---")

    try:
        slice_max = estudio.shape[0] - 1
        slice_idx_input = input(f"Ingrese el índice del corte (slice) para operar (0 a {slice_max}): ")
        if not slice_idx_input.isdigit():
            print("Por favor, ingrese solo un número entero para el índice de corte.")
            return
        slice_idx = int(slice_idx_input)
        if not (0 <= slice_idx <= slice_max):
            print("Índice de slice fuera de rango.")
            return

        print("\n--- Sub-Menu de estudio ---")
        print("a. Hacer Zoom")
        print("b. Hacer Segmentacion")
        print("c. Hacer transformacion morfologica")
        print("d. Volver al menu principal")

        op_estudio = input("Seleccione una opción (a-d): ").lower()

        if op_estudio == 'a':
            print("--- Método Zoom ---")
            x1_def, y1_def, x2_def, y2_def = 50, 30, 250, 200
            print(f"Usando coordenadas de ejemplo: ({x1_def},{y1_def}) a ({x2_def},{y2_def})")

            # Pedimos el nombre aquí, para pasarselo a la funcion
            nombre_salida_zoom = input("Ingrese nombre para guardar la imagen recortada (ej. recorte.png): ")
            estudio.metodo_zoom(slice_idx, x1_def, y1_def, x2_def, y2_def, nombre_salida_zoom)

        elif op_estudio == 'b':
            print("--- Funcion segmentacion ---")
            print("  0: BINARY, 1: BINARY_INV, 2: TRUNC, 3: TOZERO, 4: TOZERO_INV")
            tipo_bin = input("Elija el tipo (0-4): ")
            estudio.funcion_segmentacion(slice_idx, tipo_bin)

        elif op_estudio == 'c':
            print("--- Transformacion morfologica ---")
            k_size = int(input("Ingrese el tamaño del kernel (ej. 3, 5, 7): "))
            estudio.transformacion_morfologica(slice_idx, k_size, operacion='erosion')

        elif op_estudio == 'd':
            return

    except Exception as e:
        print(f"Error en la operacion del estudio: {e}")

def main():
    while True:
        mostrar_menu_principal()
        opcion = input("Seleccione una opción (1-5): ")

        if opcion == '1':
            ruta = input("Ingrese la ruta de la carpeta DICOM: ")
            if not os.path.isdir(ruta):
                print("Error: La ruta no existe.")
                continue
            try:
                print("Cargando y reconstruyendo en 3D...")
                gestor = GestionDICOM(ruta)
                estudio = EstudioImaginologico(gestor)
                lista_estudios.append(estudio)
                print(f"Estudio cargado. {len(lista_estudios)} estudios en memoria.")
                print(f"  Shape del volumen 3D: {estudio.shape}")

                # Esta función guarda la imagen, no la muestra
                gestor.mostrar_vistas_3d()

            except Exception as e:
                print(f"Error al cargar la carpeta DICOM: {e}")

        elif opcion == '2':
            if not lista_estudios:
                print("Primero debe cargar un estudio (Opción 1).")
                continue
            estudio_activo = lista_estudios[-1]
            menu_funciones_estudio(estudio_activo)

        elif opcion == '3':
            if not lista_estudios:
                print("Primero debe cargar un estudio (Opción 1).")
                continue
            estudio_activo = lista_estudios[-1]
            nombre_salida = input("Ingrese el nombre del archivo NIFTI de salida (ej. sarcoma.nii.gz): ")
            if not (nombre_salida.endswith('.nii') or nombre_salida.endswith('.nii.gz')):
                print("Error: El archivo de salida debe tener extensión .nii o .nii.gz")
                continue
            try:
                estudio_activo.gestor.convertir_a_nifti(nombre_salida)
            except Exception as e:
                print(f"Error al convertir a NIFTI: {e}")

        elif opcion == '4':
            if not lista_estudios:
                print("No hay estudios cargados para guardar.")
                continue
            lista_dfs = [est.generar_dataframe() for est in lista_estudios]
            df_completo = pd.concat(lista_dfs, ignore_index=True)
            nombre_csv = "informacion_estudios.csv"
            df_completo.to_csv(nombre_csv, index=False)
            print(f"Información de {len(lista_estudios)} estudios guardada en '{nombre_csv}'")

        elif opcion == '5':
            print("Saliste del sistema.")
            break

        else:
            print("Opcion incorrecta. Intente de nuevo.")

if __name__ == "__main__":
    main()