@echo off
Rem Setup for Windows

Rem Make a python environment if it is not present
if exist venv\ (
  echo Environment exists, so skipping to Activation
) else (
  echo Initializing new Python virtual environment
  
  python -m venv venv\
)

Rem Activate the virtual environment
echo Activating virtual environment

venv\Scripts\activate.bat

Rem Install dependencices
echo Installing dependencies

pip install -r requirements.txt

echo Everything done!
echo Run 'venv\Scripts\activate.bat' to activate your environment the next time you open a terminal to start working
echo Run 'jupyter notebook' to start the notebook after activation