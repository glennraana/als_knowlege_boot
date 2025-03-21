# ALS Knowledge Norway

An AI-powered knowledge base and chatbot for ALS patients and caregivers in Norway. This application allows users to:

1. Ask questions about ALS and receive informed responses
2. Contribute their experiences, symptoms, and solutions
3. Upload documents, images, and information about aids/equipment
4. Search for aids and solutions based on specific problems or symptoms

## Features

- **RAG (Retrieval Augmented Generation)** - Combines document retrieval with LLM generation
- **MongoDB Integration** - Stores user contributions and vector embeddings
- **Multi-format Support** - Process text, PDFs, and images
- **User-friendly Interface** - Simple forms for knowledge contributions
- **AI-powered Chat** - Answer questions with context from the knowledge base
- **Vector Search** - MongoDB Atlas vector search for semantic similarity matching of documents

## Setup

1. Clone this repository
2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Install dependencies:
   ```
   # Activate the virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install requirements
   pip install -r requirements.txt
   ```
4. The MongoDB connection is already configured with your provided connection string
5. Run the application:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `rag/`: Contains RAG implementation
  - `embeddings.py`: Embedding functionality
  - `vectorstore.py`: MongoDB Atlas vector search integration
  - `retriever.py`: Document retrieval logic
- `db/`: Database models and operations
  - `models.py`: Data models
  - `operations.py`: MongoDB database operations
- `document_processor/`: Document processing utilities
  - `processor.py`: Process text, PDFs, and images
- `data/`: Storage for temporary documents

## MongoDB Integration

This project uses MongoDB Atlas for storing:
- User contributions (problems, aids, medicine info)
- File attachments (using GridFS)
- Vector embeddings for semantic search

The MongoDB connection is pre-configured with the connection string:
```
mongodb+srv://Cluster80101:VXJYYkR6bFpL@cluster80101.oa4vk.mongodb.net/als_data
```

## Recent Updates

### Vector Search Improvements

The vector search functionality has been improved with the following features:

- **Dimension Handling** - Automatically adjusts embedding dimensions to match the database (padding/truncating)
- **Error Handling** - Robust error handling with detailed diagnostics
- **Fallback Mechanisms** - Graceful degradation when vector search issues occur
- **MongoDB Atlas Integration** - Proper syntax for MongoDB Atlas vector search using `knnBeta` operator

### Testing

A comprehensive test script (`test_vector_search.py`) is included to validate vector search functionality:

```bash
python test_vector_search.py
```

This test script verifies:
- Embedding model initialization
- Embedding generation
- Vector search index availability
- Dimension handling
- Query execution with different embedding dimensions

## Deployment to Streamlit Cloud

For å deploye applikasjonen til Streamlit Cloud:

1. Opprett en konto på [Streamlit Cloud](https://streamlit.io/cloud) hvis du ikke allerede har en
2. Klikk på "New app" i Streamlit Cloud dashboardet
3. Velg GitHub-repositoriet hvor ALS Knowledge-koden er lagret
4. Sett opp følgende:
   - **Repository**: Velg ditt repository
   - **Branch**: main (eller master)
   - **Main file path**: app.py
5. Under "Advanced settings":
   - Legg til alle hemmeligheter fra `.env` filen i "Secrets"-seksjonen
   - Dette inkluderer OPENAI_API_KEY, MONGODB_URI og MONGODB_DB_NAME
   - Format:
     ```
     [secrets]
     OPENAI_API_KEY = "din-api-nøkkel"
     MONGODB_URI = "mongodb+srv://..."
     MONGODB_DB_NAME = "als_knowledge"
     ```
6. Klikk "Deploy!"
7. Etter deployment, vil appen være tilgjengelig på en URL som `https://[username]-[app-name].streamlit.app`

### Viktige notater for Streamlit Cloud

- **Sikkerhet**: Aldri legg inn API-nøkler eller hemmelige verdier direkte i koden
- **Ytelse**: Streamlit Cloud har begrensninger på minne og CPU-bruk
- **Inaktivitet**: Apper som ikke brukes vil gå i dvale etter en time
- **Oppdateringer**: Appen vil automatisk oppdateres når du pusher nye endringer til repositoriet

### Periodisk vedlikehold

MongoDB Atlas krever periodisk vedlikehold:
- Sjekk databasen for lagringsbruk
- Rengjør unødvendige eller duplikate embeddings regelmessig
- Overvåk bruk av OpenAI API-nøkkelen og kostnadene

## Contributing

This project is designed to grow with community contributions. Users can add their knowledge directly through the application interface.

## License

MIT
