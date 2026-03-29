@echo off
echo Checking and installing required dependencies...
pip install -r requirement.txt
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies!
    exit /b 1
)
 
where python
python --version
 
REM Step 2: Run model downloader
echo Running model downloader...
python modeldownloader.py
IF %ERRORLEVEL% NEQ 0 (
    echo Model downloader failed!
    exit /b 1
)
 
REM Step 3: Navigate to Query directory
cd Query
IF %ERRORLEVEL% NEQ 0 (
    echo Query directory not found!
    exit /b 1
)
 
REM Step 4: Start FastAPI app
echo Starting FastAPI app...
start /B uvicorn QueryProcessing:app --host 0.0.0.0 --port 8088
 
REM Wait a few seconds for server to start (no curl check here)
echo Waiting for FastAPI to be available...
timeout /t 5 /nobreak >nul
 
REM Step 5: Navigate to FrontEndDesign
cd ..\FrontEndDesign
IF %ERRORLEVEL% NEQ 0 (
    echo FrontEndDesign directory not found!
    exit /b 1
)
 
REM Step 6: Open index.html in default browser
echo Opening index.html in your default browser...
start "" index.html
 
REM Step 7: Keep script running
echo FastAPI running in background. Press any key to exit...
pause >nul