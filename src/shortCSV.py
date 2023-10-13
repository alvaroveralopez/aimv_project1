import pandas as pd

# 1. Leer el archivo CSV en un DataFrame
csv_file = "../database.csv"  # Reemplaza "tu_archivo.csv" con la ruta de tu archivo CSV
df = pd.read_csv(csv_file)

# 2. Ordenar el DataFrame por la columna deseada (por ejemplo, 'nombre')
df = df.sort_values(by='name', ascending=True)  # Cambia 'nombre' al nombre de la columna por la que deseas ordenar

# 3. Guardar el DataFrame ordenado en un nuevo archivo CSV en el mismo directorio
df.to_csv(csv_file, index=False)  # Guardar el DataFrame en el nuevo archivo

print(f"El archivo CSV ha sido ordenado y guardado como '{csv_file}' en el mismo directorio.")
