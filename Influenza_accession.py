from Bio import Entrez

def fetch_influenza_accession_numbers():
    # Set your email for Entrez (required by NCBI)
    Entrez.email = "your_email@example.com"
    
    # Search for human Influenza genome sequences
    search_query = "Influenza[Title] AND Homo sapiens[Organism] AND genome"
    
    # Perform the search in the Nucleotide database
    handle = Entrez.esearch(db="nucleotide", term=search_query, retmax=20)
    record = Entrez.read(handle)
    handle.close()
    
    # Fetch the accession numbers
    accession_numbers = record['IdList']
    return accession_numbers

# Get the accession numbers
influenza_accessions = fetch_influenza_accession_numbers()
print("Human Influenza Accession Numbers:", ",".join(influenza_accessions))
