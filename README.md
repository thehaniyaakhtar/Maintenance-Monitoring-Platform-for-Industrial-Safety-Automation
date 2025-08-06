# Maintenance Monitoring Platform for Industrial Safety Automation

---

## Overview

**Maintenance Monitoring Platform for Industrial Safety Automation** is a centralized web platform designed to streamline and optimize the maintenance processes of industrial machinery.  
It helps industries by scheduling preventive maintenance, generating timely alerts based on issue severity, and monitoring machine performance data.

By integrating technology and data-driven solutions, this system aims to **enhance operational efficiency**, **reduce unexpected downtimes**, and **improve equipment lifespan**.

Users can select a machine, input real-time operational parameters (like Air Temperature, Process Temperature, Rotational Speed, Torque, and Tool Wear), and the system predicts possible machine faults and recommends preventive actions.

---

## Features

- **Preventive Maintenance Scheduling:**  
  Select a machine, choose a severity level, describe issues, and schedule maintenance with reminders.

- **Dynamic Alert System:**  
  Automatically generate alerts based on the described issues and selected severity.

- **Reminder Notifications:**  
  Set a date and time for future reminders; pop-up notifications are triggered both immediately after submission and at the scheduled time.

- **User Management:**  
  Signup and Login system for access control.
  
- **Dashboard View:**  
  Future-ready space for displaying real-time machine data and maintenance statistics.

- **Responsive and User-Friendly UI:**  
  Consistent design across all pages with well-spaced input fields, interactive dropdowns, and proper form validations.

---

## Project Structure

INDUSTRY/

├── node_modules/            
├── public/
│   ├── __pycache__/        
│   ├── dataset.csv         
│   ├── data.json            
│   ├── home.jpg             
│   ├── index.html           
│   ├── industry.html        
│   ├── login.html           
│   ├── signup.html        
│   ├── maintanence.html    
│   ├── alert.html           
│   ├── dashboard.html       
│   ├── styles.css           
│   ├── server.py  
│   ├── model.py

├── venv/

├── machine_data.db

├── package.json

├── server.js

└── requirements.txt         

---

## Technologies Used

- **Frontend:**
  - HTML5, CSS3, JavaScript
- **Backend:**
  - Python (Flask)
  - (Optional) Node.js for alternate server routes
- **Database:**
  - SQLite3 / CSV + JSON-based simple storage
- **Tools:**
  - Visual Studio Code
  - GitHub
- **Libraries:**
  - Python modules: `flask`, `json`, `sqlite3`
  - JavaScript: DOM manipulation, event handling

---

## How to Run Locally

1. **Clone the Repository**

   ```bash
   git clone https://github.com/thehaniyaakhtar/industrial-maintenance-hub.git
   cd industrial-maintenance-hub
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/macOS)
   .\venv\Scripts\activate    # (Windows)
   ```

3. **Install Python Requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask Server**

   ```bash
   python server.py
   ```

5. **Open Frontend**

   - Navigate to `localhost:5000` (or your server's IP) in your browser.
   - Explore all the features: maintenance scheduling, alerts, reminders, dashboard, etc.

---

## Screenshots
  -Main Home Page
![Screenshot (299)](https://github.com/user-attachments/assets/185ce475-91d8-4161-9697-fcd6f7708012)

  -Sign Up Page
![Screenshot (300)](https://github.com/user-attachments/assets/21fee757-eb83-4b9b-8832-500959303db7)

 -Machine Failure Prediction
![Screenshot (309)](https://github.com/user-attachments/assets/6f4f563f-8ab1-448e-9aa2-8d44f1db5c07)
  
  -Evaluation Metrics
![Screenshot (310)](https://github.com/user-attachments/assets/0cbd78d6-0e6f-41cd-ad5f-a6dc1f8999a6)

  -Comparison for Actual and Predicted RUL
![Screenshot (311)](https://github.com/user-attachments/assets/95acee38-380b-4b2f-aa28-6b7531731ebd)

  -Scheduling a Maintainence
![Screenshot (305)](https://github.com/user-attachments/assets/3c60cabe-88cd-4641-9c91-d6a792ac4657)

![Screenshot (306)](https://github.com/user-attachments/assets/4fb0c4f7-2885-413d-8837-c54a91b46966)

  -Setting a notification for maintainence/previously scheduled
![Screenshot (302)](https://github.com/user-attachments/assets/1d122284-0f59-431b-b0b7-369b1380d38a)

  -Set Notification
![Screenshot (303)](https://github.com/user-attachments/assets/ce592e76-f841-431c-a330-addf18184dcd)

  -Notification from Scheduled time
![Screenshot (304)](https://github.com/user-attachments/assets/5ffb0016-fcab-4688-bb63-600dd52ac630)

  -Recording values from past observation for  all parameters
![Screenshot (307)](https://github.com/user-attachments/assets/4867b17c-f4cc-4648-b17a-fbf3dc0b6edd)

  -Checking comparison of all past values for selected paramater
![Screenshot (308)](https://github.com/user-attachments/assets/4f4c8b3d-3ae4-4c13-9988-4f3abbdac15e)

---

## Future Scope

- Integration with IoT sensors for real-time machine data.
- Multi-user role-based access (Admin, Maintenance Staff, Manager).
- Graphical dashboard with maintenance trends and KPIs.

---

## License

  -This project is for academic purposes. Feel free to fork and customize it!

---
