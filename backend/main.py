import uvicorn
import os

if __name__ == "__main__":
    # Run locally
    #uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

    # Run on Heroku
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("app.api:app", host='0.0.0.0', port=port)
