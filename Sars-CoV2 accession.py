from Bio import Entrez

def fetch_covid_accession_numbers():
    # Set your email for Entrez (required by NCBI)
    Entrez.email = "your_email@example.com"
    
    # Search for SARS-CoV-2 genome sequences
    search_query = "SARS-CoV-2[Organism] AND genome"
    
    # Perform the search in the Nucleotide database
    handle = Entrez.esearch(db="nucleotide", term=search_query, retmax=200)
    record = Entrez.read(handle)
    handle.close()
    
    # Fetch the accession numbers
    accession_numbers = record['IdList']
    return accession_numbers

# Get the accession numbers
covid_accessions = fetch_covid_accession_numbers()
print("Covid (SARS-CoV-2) Accession Numbers:", ",".join(covid_accessions))
