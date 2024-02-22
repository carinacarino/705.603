class ETL_Pipeline:
    """Extracts data from a csv file, transforms features, and load transformed data to a new csv file"""
    
    def extract(filename):
        """
        Extract data from a csv file and returns it as a Pandas DataFrame.
        
        """
                try:
            data = pd.read_csv(filename)
            return data
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None