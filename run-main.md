how to run the main file , 
 execute this simple line of code to the terminal and main wil run
        uvicorn main:app --host 0.0.0.0 --port 8000
        uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600

after run goto browser for verification : http://localhost:8000/docs