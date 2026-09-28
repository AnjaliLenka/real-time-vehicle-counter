# Real-Time Vehicle Counter — Streamlit App

A Streamlit version of the YOLO11n + Object Tracking vehicle counter project.

## Folder structure

```text
Vehicle_Counter_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── yolo11n.pt              <- PUT YOUR MODEL HERE
├── models/
│   └── PUT_MODEL_HERE.txt
└── videos/
    └── PUT_TRAFFIC_VIDEO_HERE.txt
```

## 1. Add the model

Copy the `yolo11n.pt` from your completed Jupyter project into the project root, next to `app.py`.

## 2. Install

Open Command Prompt in this folder:

```cmd
python -m pip install -r requirements.txt
```

## 3. Run locally

```cmd
python -m streamlit run app.py
```

Open the URL shown in the terminal, normally `http://localhost:8501`.

## 4. Use the app

1. Upload a traffic video.
2. Click **Process Video**.
3. Wait for processing.
4. View vehicle counts.
5. View the processed video.
6. Download the processed video.

Detected/countable classes: car, motorcycle, bus, truck.

The counting line is automatically placed at the middle of the video. A tracked vehicle is counted when its center moves from above the line to below it. Each tracking ID is counted once.

## 5. Deploy

Push these files to GitHub:

```text
app.py
requirements.txt
yolo11n.pt
```

Then create a Streamlit Community Cloud app, select the GitHub repository, `main` branch, and `app.py` as the entrypoint.

Do not upload a large traffic video just for deployment. Users upload videos through the app.

## Troubleshooting

If Streamlit is not recognized, use:

```cmd
python -m streamlit run app.py
```

If `yolo11n.pt` is missing, place it beside `app.py`.

If deployment reports a dependency/Python error, check the Streamlit deployment logs and adjust package versions if required.

## Viva summary

YOLO11n is a pre-trained object-detection model. Object tracking assigns persistent IDs to detected vehicles. The application uses those IDs to determine when vehicles cross the counting line. The final output is a processed video containing bounding boxes, class labels, tracking IDs, a counting line, and vehicle counts.

The project performs inference and application-level processing; it does not train YOLO11n from scratch.
