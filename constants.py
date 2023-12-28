chat = {
    "TITLE1": "About Me",
    "TITLE2": "AI Chat",
    "menu_v": {
   "container": {"background-color": "#a7c5ff"},
   "icon": {"color": "white", "font-size": "25px"}, 
   "nav-link": {"font-size": "25px", "text-align": "center", "color": "white", "margin":"0px", "--hover-color": "#1b97ff"},
   "nav-link-selected": {"background-color": "#1b97ff"}
    },
    "menu_h": {
        "container": {"padding": "0px",
                      "display": "grid",
                      "margin": "0!important",
                      "background-color": "#212121"
                      },
        "icon": {"color": "#bd93f9", "font-size": "14px"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "auto",
            "background-color": "#262626",
            "height": "30px",
            "width": "13rem",
            "color": "#edff85",
            "border-radius": "5px"
        },
        "nav-link-selected": {
            "background-color": "#212121",
            "font-weight": "300",
            "color": "#f7f8f2",
            "border": "1px solid #e55381"
        }
    }
}

info = {
   "Pronoun": "his",                  # -> "her"
   "Subject": "he",                   # -> "she"
   "Name": "Diego",                   # -> "Vicky"
   "Full_Name":"Diego H. Salazár A.", # -> "Vicky Kuo"
   "Intro": """Tech Passionated, IBM and Stanford Online, Professional Trained AI Enthusiast, MSc. Applied Physics in Space Apps (Balseiro), BSc. Physics Engineering (Instrumentation, 
               Control and Computing), Economy and Project Management in Energy (Balseiro), Piano and Synths (Young Musician Conservatory Degree), and more...""", 
   "About":"""Hello, I'm Diego, and I'm passionate about leveraging AI, sensors and technology to drive meaningful insights and solutions. You can explore my resume in english,                                                                
              the completed portfolio of labs and projects, courses summary, and my current AI training at https://diegsalaza.wixsite.com/miportalweb?lang=en """,
   "Projects":"https://github.com/DiegoHernanSalazar/DiegoHernanSalazarAlarcon/releases",
   "LinkedIn":"https://www.linkedin.com/in/diegohernansalazaralarcon/",
   "City":"Popayán, Colombia",
   "Resume": "https://www.linkedin.com/in/diegohernansalazaralarcon/overlay/1635554084541/single-media-viewer/?profileId=ACoAAB75w_IBPDUwB6wDyXrzBqgalFVIzwH1s2Y",
   "Email": "diegsalaza@gmail.com" 
}

projects = [
        
        {
            "title": "Building Systems with the ChatGPT API (Doing or running)",
            "description": """Automate complex workflows using chain calls to a large language model. Unlock new development capabilities and improve your efficiency,
                              build chains of prompts that interact with the completions of prior prompts, build systems where Python code interacts with both 
                              completions and new prompts, build a customer service chatbot using all the techniques from this course.
                              Apply these skills to practical scenarios, including classifying user queries to a chat agent’s response, 
                              evaluating user queries for safety, and processing tasks for chain-of-thought, multi-step reasoning.""",
            "image_url": """https://global.discourse-cdn.com/openai1/original/3X/f/d/fdc43d9561048387ab838a2ccd4d4044cf524831.jpeg""",
            "link": "https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/"                  
        },
        {
            "title": "Land Your Dream Job: Build Your Portfolio with Generative AI",
            "description": """Boost your career visibility with an AI-driven chatbot and interactive app, enabling recruiters to directly "talk" to your skills and accomplishments.
                              Create an interactive app to showcase your own data work, achievements, and personality. Then, pair that with an AI-powered chatbot where recruiters
                              can talk to your resume!. Implement a custom AI chatbot powered by GenAI from watsonx, Customize the contents of a portfolio website template.
                              A fundamental understanding of Python, HTML, CSS, and JavaScript is beneficial.""", 
            "image_url": """https://author.skills.network/rails/active_storage/representations/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBdEV3IiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--c06e7fd984c39997b5284064961cfc4ef274ee79/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdDRG9MWm05eWJXRjBTU0lJWjJsbUJqb0dSVlE2QzJ4dllXUmxjbnNHT2dadWFmbzZGSEpsYzJsNlpWOTBiMTlzYVcxcGRGc0hhUUlBQkdrQ0FBTT0iLCJleHAiOm51bGwsInB1ciI6InZhcmlhdGlvbiJ9fQ==--548c96cd3b348a6f920ac890f6c6e675e061fe51/ezgif.com-video-to-gif.gif""",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+GPXX0UVUEN+v1#about-course"
        },
        {
            "title": "(OpenAI / DeepLearning.AI). ChatGPT Prompt Engineering for Developers. Isa Fulford - Andrew Ng, Python",
            "description": """Describe how LLMs work, provide best practices for prompt engineering, use OpenAI API in applications in a variety of tasks including:  
                              Summarizing (e.g., summarizing user reviews for brevity), Inferring (e.g., sentiment classification, topic extraction), 
                            ​  Transforming text (e.g., translation, spelling & grammar correction), Expanding (e.g., automatically writing emails),
                              Two key principles for writing effective prompts, how to systematically engineer good prompts, and also learn to build a custom chatbot""",
            "image_url": """https://static.wixstatic.com/media/f95dba_344b8f095c774b09a8620dca4e770ad3~mv2.png/v1/crop/x_0,y_0,w_1625,h_691/fill/w_1225,h_521,al_c,q_95,enc_auto/ChatGPT%20Prompt%20Engineering%20for%20Developers.png""",
            "link": "https://diegsalaza.wixsite.com/miportalweb/copia-de-stanford-online-machine-lear-1?lang=en"
        },
        {
            "title": "(Google/Stanford)-Udacity. Artificial Intelligence for Robotics, Programming a Robotic Car. Sebastian Thrun, Python",
            "description": """Online SLAM, Implementing SLAM, Confident Measurements ( Z2 * (1/sigma) ), Expand (To include ONE landmark), OMEGA and xi,
                              Fun with Parameters (Twiddle-PD), Segmented CTE, Racetrack Control, Constrained Smoothing, Cyclic Smoothing, Parameters Optimization (Twiddle - PID),
                              (PID) Controller, (PD) Controller, Implement (P) Controller, Path Smoothing, Stochastic Motion, Left Turn Policy, Optimal Policy, Value Program,
                              A *, Print Path, Expansion Grid expand list, First Search Finding Optimal Robot Path, Particle Filters Final Quiz, Sensing, Circular Motion,
                              Error, Orientation 2, Resampling Wheel, New Particle, Importance Weight, Robot Particles, Creating Particles, Add Noise, Moving Robot,
                              Kalman Filters Programming Exercise, Kalman Matrices, Kalman Filter Code, Predict Function, New Mean and Variance, Maximize Gaussian,
                              Localization Program, Sense and Move 2, Sense and Move, Move 1000, Move Twice, Inexact Move Function, Move Function, Multiple Measurements,
                              Test Sense Function, Normalized Sense Function, Sense Function, Sum of Probabilities, pHit and pMiss, Generalized Uniform Distribution,
                              Uniform Distribution""",
            "image_url": """https://static.wixstatic.com/media/f95dba_690cd0074f2c442c99cc2410012d14b4~mv2.png/v1/crop/x_0,y_0,w_1500,h_558/fill/w_1191,h_444,al_c,q_95,enc_auto/AI%20for%20Robotics%20Main.png""",
            "link": "https://diegsalaza.wixsite.com/miportalweb/copia-de-stanford-online-machine-lear?lang=en"
        },      
        {
            "title": "IBM AI Engineering Professional Certificate (V2)",
            "description": """Data Science, Python Libraries, Machine Learning, Regression, Hierarchical Clustering, K-Means Clustering, Deep Learning, Artificial Neural Networks,
                              Artificial Intelligence AI, OpenCV, Image Processing, Computer Vision, Keras, Pytorch, TensorFlow, Several ML & DL projects and he is now armed 
                              with skills for starting a career in AI Engineering""",
            "image_url": """https://static.wixstatic.com/media/f95dba_0772032d11314b44a27410c242915966~mv2.jpg/v1/fill/w_555,h_555,al_c,q_90,enc_auto/Professional_Certificate_-_AI_Engineerin.jpg""",
            "link": "https://diegsalaza.wixsite.com/miportalweb/copia-de-ibm-data-science-professiona?lang=en"
        },
        {
            "title": "IBM Data Science Professional Certificate (V2)",
            "description": """Data Science, Data Analysis, Data Visualization, Machine Learning ML, Python, SQL, Database, Jupyter, Notebook, Artificial Intelligence AI, Watson 
                              Studio, IBM Cloud, Db2, Pandas, Numpy, Bokeh, Matplotlib, Folium, Seaborn, Scikit-learn, SciPy, RStudio, Zeppelin, Regression, Clustering, 
                              Classification, Location, Methodology, Foursquare, Recommender Systems""",  
            "image_url": """https://static.wixstatic.com/media/f95dba_e2cccaa959ce4d75a660d4d8b5ab01c9~mv2.jpg/v1/fill/w_555,h_555,al_c,q_90,enc_auto/Professional_Certificate_-_Data_Science_.jpg""",
            "link": "https://diegsalaza.wixsite.com/miportalweb/about-4?lang=en"
        },
        {
            "title": "Stanford University Machine Learning",
            "description": """Broad introduction to machine learning, datamining, and statistical pattern recognition. Supervised learning (parametric/non-parametric algorithms,
                              support vector machines, kernels, neural networks). Unsupervised learning (clustering, dimensionality reduction, recommender systems, deep learning).
                              Best practices in machine learning (bias/variance theory; innovation process in machine learning and AI). Smart robots (perception, control),
                              text understanding (web search, anti-spam), computer vision, medical informatics, audio, database mining, and other areas""",  
            "image_url": "https://static.wixstatic.com/media/f95dba_df9cdb8051974c3f961492132aef6a82~mv2.jpg/v1/fill/w_363,h_363,al_c,lg_1,q_90,enc_auto/large-icon_edited.jpg",
            "link": "https://diegsalaza.wixsite.com/miportalweb/copia-de-ibm-ai-engineer-professional?lang=en"
        },
        {
            "title": "Applied Data Science Capstone Project",
            "description": """Complete the Machine Learning Prediction lab, Space X Falcon 9 First Stage Landing Prediction (Assignment: Machine Learning Prediction), 
                              Build a Dashboard Application with Plotly Dash, Complete the Interactive Visual Analytics with Folium lab (Launch Sites Locations Analysis with Folium),
                              Complete de Exploratory Data Analysis (EDA) with Visualization ( SpaceX Falcon 9 First Stage Landing Prediction Assignment: Exploring and Preparing 
                              Data), Complete the Exploratory Data Analysis (EDA) with SQL Magic and Db2 Connection at Python Notebook (Assignment: SpaceX SQL Notebook for Peer 
                              Assignment), Space X Falcon 9 First Stage Landing Prediction Lab 2: Data wrangling (EDA Exploratory Data Analysis),
                              Space X Falcon 9 First Stage Landing Prediction (Web scraping Falcon 9 and Falcon Heavy Launches Records from Wikipedia),
                              SpaceX Falcon 9 first stage Landing Prediction Lab 1: Collecting the data""",
            "image_url": """https://static.wixstatic.com/media/f95dba_401153b40dec4adfb9839e0295cc742e~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/DSc_Capstone_Certificate_edited_edited_j.jpg""",
            "link": "https://www.credly.com/badges/a6c6904c-d6dc-4d81-894a-52f45f464602/linked_in_profile"
        },
        {
            "title": "AI Capstone Project with Deep Learning",
            "description": """VGG16 Pre-trained Model with Keras, compared with ResNet50 stored model with Keras (Final Assignment), 
                              Pre-trained-Model ResNet18 with PyTorch (Final Assignment), Pre-Trained Models (Keras), Linear Classifier with PyTorch, Data Preparation with Keras,
                              Data Preparation with PyTorch, Loading Data (Keras), Loading Data (PyTorch)""",
            "image_url": """https://static.wixstatic.com/media/f95dba_0f1a97bdb23f47b18e8bd0fe09372e15~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/AI_Capstone_Certificate_edited.jpg""",
            "link": "https://www.credly.com/badges/51a84ecd-c0d7-45b7-a1a6-e8a378a81a6c/linked_in_profile"
        },
        {
            "title": "Building Deep Learning Projects with TensorFlow",
            "description": """AUTOENCODERS, RESTRICTED BOLTZMANN MACHINES, RECURRENT NETWORKS and LSTM IN DEEP LEARNING, RECURRENT NETWORKS IN DEEP LEARNING, 
                              CONVOLUTIONAL NEURAL NETWORK APPLICATION, LOGISTIC REGRESSION WITH TENSORFLOW, LINEAR REGRESSION WITH TENSORFLOW, TENSORFLOW'S HELLO WORLD",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/diamond.png""",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+GPXX0FTNEN+v1"
        },
        {
            "title": "Intro to Computer Vision and Image Processing",
            "description": """Final Project Stop and Non-Stop Signs Classification for Self-Driving Cars Start-up. Transfer Learning with Convolutional Neural Networks For
                              Classification with PyTorch and Computer Vision Learning Studio (CV Studio),
                              Object Detection with Convolution Neural Network (CNN) based on Tensorflow (CV Studio), Object detection with Faster R-CNN,
                              Car Detection with Haar Cascade Classifier (CV Studio), Transfer Learning with Convolutional Neural Networks For Classification with PyTorch and Computer
                              Vision Learning Studio (CV Studio), Data Augmentation, Convolutional Neural Network, Training A Neural Network with Momentum, Neural Network Rectified
                              Linear Unit (ReLU) vs Sigmoid, Practice: Neural Networks with One Hidden Layer: Noisy XOR,  Image Classification with HOG and SVM,
                              H.O.G. and SVM Image Classification with OpenCV and Computer Vision Learning Studio (CV Studio), Support Vector Machine (SVM) vs Vanilla Linear 
                              Classifier, Hand-Written Digits Image Classification with Softmax, Logistic Regression With Mini-Batch Gradient Descent, KNN Image Classification with 
                              OpenCV and Computer Vision Learning Studio (CV Studio), Spatial Filtering with OpenCV, Spatial Filtering Operations with Pillow, Geometric Operations and 
                              Other Mathematical Tools with OpenCV, Geometric Operations and Other Mathematical Tools with Pillow, Histogram and Intensity Transformations,
                              Manipulating Images with OpenCV, Manipulating Images with PIL, OpenCV Library, Pillow Library (PIL)""",
            "image_url": """https://static.wixstatic.com/media/f95dba_fb42a0db2ee548349f629b2cf32ad5bb~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certificate_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/7f1e63ab-7024-41ac-94bd-401131634195/linked_in"      
        },
        {
            "title": "Deep Neural Networks with PyTorch",
            "description": """Fashion-MNIST Project, Convolutional Neural Network with Batch-Normalization, Convolutional Neural Network MNIST with Small Images, 
                              Convolutional Neural Network Simple example, Multiple Input and Output Channels (Convolution), Activation function and Maxpooling, 
                              WHAT´S CONVOLUTION, Batch Normalization with the MNIST Dataset, Neural Networks with MomentumNeural Networks with Momentum, Momentum,
                              Test Uniform, Default and He Initialization on MNIST Dataset with Relu Activation, Test Uniform, Default and Xavier Uniform Initialization on MNIST 
                              dataset with tanh activation, Initialization with Same Weights, Using Dropout in Regression, Using Dropout for Classification, Deeper Neural Networks 
                              with nn.ModuleList(), Hidden Layer Deep Network: Sigmoid, Tanh and Relu Activations Functions MNIST Dataset, Test Sigmoid, Tanh, and Relu Activations 
                              Functions on the MNIST Dataset, Activation Functions, Neural Networks with One Hidden Layer: Multiple Outputs (Multiclass), Neural Networks with One   
                              Hidden Layer: Multiple Outputs (Multiclass), Practice: Neural Networks with One Hidden Layer: Noisy XOR, Neural Networks More Hidden Neurons, Simple One  
                              Hidden Layer Neural Network, Softmax Classifier 2D, Softmax Classifer 1D, Logistic Regression Training Negative Log likelihood (Cross-Entropy), Logistic  
                              Regression and Bad Initialization Value, Logistic Regression, Linear Regression Multiple Outputs, Multi-Target Linear Regression, Linear Regression 
                              Multiple Outputs,Multiple Linear Regression, Linear regression: Training and Validation Data, Linear Regression 1D: Training Two Parameter Mini-Batch 
                              Gradient Descent, Linear Regression 1D: Training Two Parameter Mini-Batch Gradient Decent, Linear regression 1D: Training Two Parameter Stochastic  
                              Gradient Descent (SGD), Linear regression 1D: Training Two Parameter, Linear Regression 1D: Training One Parameter, Linear Regression 1D: Prediction,  
                              Prebuilt Datasets and Transforms, Image Datasets and Transforms, Simple Dataset, Differentiation in PyTorch, Two-Dimensional Tensors, Torch Tensors in
                              1D.""", 
            "image_url": """https://static.wixstatic.com/media/f95dba_9a53f150d9a54f53a3df12eedc9ba8e6~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certificate_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/22607f56-e06e-440d-a231-93aaa463a706"
        },
        {
            "title": "Intro to Deep Learning and Neural Networks with Keras",
            "description": """FINAL PROJECT DEEP LEARNING & NEURAL NETWORKS WITH KERAS, Convolutional Neural Networks with Keras, Classification Models with Keras,
                              Regression Models with Keras, Artificial Neural Networks - Forward Propagation""",
            "image_url": """https://static.wixstatic.com/media/f95dba_fc9223c8cd9f49878ab6c6bcbd3e0589~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certificate_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/96f812e4-495f-40d0-b592-c7d042296912?source=linked_in_profile"
        },
        {
            "title": "Machine Learning with Python",
            "description": """Final Project Machine Learning with Python (Classification with Python), COLLABORATIVE FILTERING, CONTENT-BASED FILTERING, Density-Based Clustering,
                              Hierarchical Clustering, K-Means Clustering, SVM (Support Vector Machines), Logistic Regression with Python, Decision Trees, K-Nearest Neighbors,
                              Non Linear Regression Analysis, Polynomial Regression, Multiple Linear Regression, Simple Linear Regression""",
            "image_url": """https://static.wixstatic.com/media/f95dba_9a9c028fa5c84c3d93f7e7da62c08c78~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certificate_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/9f2356bf-b755-4345-846b-19e87d2b1495/public_url"
        },
        {
            "title": "Data Analysis with Python",
            "description": """Final Project Data Analysis with Python, Data Analysis with Python Module 5: Model Evaluation and Refinement, 
                              Data Analysis with Python Module 4: Model Development, Data Analysis with Python Exploratory Data Analysis, Data Analysis with Python Data Wrangling,
                              Data Analysis with Python Introduction, Data Analysis with Python Introduction""",
            "image_url": """https://static.wixstatic.com/media/f95dba_b32276053b41429da2c54de5af832fff~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certificate_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/ac8f06c2-2744-4576-94c9-c47fe5952465/linked_in_profile" 
        },
        {
            "title": "Data Visualization with Python",
            "description": """FINAL PROJECT DATA VISUALIZATION WITH PYTHON, Generating Maps with Python, Waffle Charts, Word Clouds, and Regression Plots, 
                              Pie Charts, Box Plots, Scatter Plots, and Bubble PlotsArea Plots, Histograms, and Bar Plots, Introduction to Matplotlib and Line Plots,
                            Analyzing US Economic Data and Building a Dashboard""",
            "image_url": """https://static.wixstatic.com/media/f95dba_fe7ba5c6fe78447d98bb703100975723~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certificate_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/aa66e32b-bb9f-4f4f-95c8-4e22a2ea9f8b/linked_in_profile"
        },
        {
            "title": "Databases and SQL for Data Science with Python",
            "description": """Assignment: Notebook for Peer Assignment, Working with a real world data-set using SQL and Python, Analyzing a real world data-set with SQL and Python,
                              Accessing Databases with SQL Magic, Access DB2 on Cloud using Python, Connect to Db2 database on Cloud using Python""",
            "image_url": """https://static.wixstatic.com/media/f95dba_f34d9c1e2a8842559414caedc25c6628~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge%20Certification_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/b798aa4b-8586-4214-b127-be9600015705/linked_in?t=ri7ssi"
        },
        {
            "title": "Python for Data Science, AI and Development",
            "description": """Application Programming Interface, A pplication Programming Interface (API), 2D Numpy in Python, 1D Numpy in Python, Introduction to Pandas Python,
                              Write and Save Files in Python, Reading Files Python, Classes and Objects in Python, Classes and Objects in Python, Functions in Python, 
                              Functions in Python, Loops in Python, Loops in Python, Conditions in Python, Conditions in Python, Sets in Python, Dictionaries in Python, 
                              Lists in Python, Tuples in Python, String Operations, Python - Writing Your First Python Code!""",
            "image_url": """https://static.wixstatic.com/media/f95dba_206521dc38f14bdd9d6841c70cf8970e~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certification_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/3b0237ca-27a5-426d-ad99-6a73e9a97a25/linked_in_profile"
        },
        {
            "title": "Python Project for Data Science",
            "description": """Extracting and Visualizing Stock Data, Extracting Stock Data Using a Web Scraping, Extracting Stock Data Using a Python Library,
                              Web Scraping Lab""",
            "image_url": """https://static.wixstatic.com/media/f95dba_f38b9d660a3342b5ab47b49aa9ec8712~mv2.jpg/v1/fill/w_750,h_580,al_c,q_90,enc_auto/Badge_Certification_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/6d5a91bd-dc23-4037-9a8e-873d611d589e/public_url"
        },
        {
            "title": "Data Science Methodology",
            "description": "From Modeling to Evaluation, From Understanding to Preparation, From Requirements to Collection, From Problem to Approach",
            "image_url": """https://static.wixstatic.com/media/f95dba_cabf459627284b62ba69917f1a49d5a6~mv2.jpg/v1/fill/w_750,h_581,al_c,q_90,enc_auto/Badge%20Certification_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/856b0f99-0af4-4182-a508-666f26e150e4/linked_in?t=rdkxwg"
        },
        {
            "title": "Tools for Data Science",
            "description": "My Jupyter Notebook on IBM Watson Studio (Tools for Organizing a Notebook via Markdown)",
            "image_url": """https://static.wixstatic.com/media/f95dba_3fff974801054a57b583b8e32fb1d9ff~mv2.jpg/v1/fill/w_750,h_581,al_c,q_90,enc_auto/Badge_Certification_edited_edited.jpg""",
            "link": "https://www.credly.com/badges/dfd33601-d42d-4cf3-8ab4-289d311a6143/linked_in?t=rbr3it"
        },
        {
            "title": "NI National Instruments Professional Certifications",
            "description": """LabVIEW Core 1 (Develop basic applications in the LabVIEW graphical programming environment, Create applications using a state machine design pattern,
                              Read and write data to file), LabVIEW Core 2 (Design, implement, and distribute stand-alone applications using LabVIEW, Apply single and multiple-loop
                              design patterns for application functionality), LabVIEW Core 3 (Follow an agile software development process to desgin, implement, document, and test key
                              application features, Learn the skills needed to create scalable, readable, and maintainable applications, Develop Successful Applications,
                              Organize Projects, Create Application Architecture, Create Professional User Interfaces, Manage and Log Errors, Create Modular Code), 
                              LabVIEW FPGA (Create & compile your LabVIEW FPGA VI and download to NI reconfigurable I/O hardware, Acquire and generate analog and digital signals,
                              control timing, synchronize operations, and implement signal processing on the FPGA, Communicate between the FPGA and a host, 
                              Design and implement applications using the LabVIEW FPGA module)""",       
        },
        {
            "title": "Design and Characterization of a Radiation Monitor for Space Applications (MARE) project",
            "description": """Parameters for detector design (Physics of interaction of radiation with matter, SBD detectors, Magnetic filters, Mechanical filters for ionizing,
                              radiation´s stopping power in a solid), Analog detection chain (Analog electronics developed as flight prototype), Prototype of digital electronics 
                              developed to elaborate the ionizing radiation spectrum, through communication with the on-board computer, via RS-232 interface, 
                              Study, develop and integration the main systems of the Argentine Space Radiation Monitor (MARE), Demonstrating detector functionality through computer
                              data collection, Engineering Tests (Tests with radiation detectors, Characterization of Analog Electronics associated to detectors,
                              Development of numerous tasks to solve Digital Electronics problems).""",
            "image_url": """https://static.wixstatic.com/media/f95dba_4b901301df4048ccb8508208c133b8d9~mv2.png/v1/fill/w_1225,h_600,al_c,q_95,enc_auto/f95dba_4b901301df4048ccb8508208c133b8d9~mv2.png""",
            "link": """https://campi.cab.cnea.gov.ar/opacmarc/cgi-bin/wxis?IsisScript=%2Fxis%2Fopac.xis&db=Falicov&searchType=TITLE&query=Dise%F1o+y+Caracterizaci%F3n+de+un+Monitor+de+Radiaci%F3n+para+Aplicaciones+Espaciales"""
        },
        {
            "title": "Manufacture and characterization of thin films of zinc oxide (ZnO), made with the RF sputtering technique, for the analysis and study of its piezoelectric properties. (New electronic device)",
            "description": """The main applications of a piezo transducer are identified at: Signal amplifiers, ultrasound sensors and ultrasonic activated switches. 
                              Manufacture and characterization of semiconductor thin films (Piezo Thin Films), Operation of magnetron sputtering RF Balzers BAE-250 system by RF
                              tuning of RLC circuit, vacuum and high vacuum systems (mechanical pump and turbo molecular turbine), Eurotherm temperature control operation.""",
            "image_url": """https://static.wixstatic.com/media/f95dba_7e2102b0ba57407daa971daf3c9d5112~mv2.png/v1/fill/w_629,h_461,al_c,lg_1,q_95,enc_auto/f95dba_7e2102b0ba57407daa971daf3c9d5112~mv2.png""",
            "link": """https://www.linkedin.com/in/diegohernansalazaralarcon/details/education/1474664384744/single-media-viewer/?profileId=ACoAAB75w_IBPDUwB6wDyXrzBqgalFVIzwH1s2Y"""     
        },
        {   "title": "Design and implementation of an electronic frequency selector system, for characterization of piezoelectric thin films. (Instrumentation, control and computing)",
            "description": """Design, electronic implementation and data acquisition with NI LabVIEW, Development of the “Frequency Selector System FSS” project, with MATLAB,
                              MPLAB (C Language), PROTEUS and LabVIEW software for design and integration of digital PI controller""",
            "image_url": """https://static.wixstatic.com/media/f95dba_4e3ab22f745f4611984b1f296aaf588c~mv2.jpg/v1/fill/w_773,h_551,al_c,q_90,enc_auto/Dise%C3%B1o_Electronico_edited.jpg""",
            "link": """https://www.researchgate.net/publication/283014950_Design_And_Implementation_Of_A_Frequency_Selector_System_For_Characterization_Of_Piezoelectric_Thin_Films"""
        },
        {
            "title": "Professional MULTI CHANNEL ANALYZER (MCA), using NI LabVIEW (National Instruments technology)",
            "description": "MCA uses NI LabVIEW and SOLIDWORKS link, for high counting and motion analysis in Solar Panel",
            "image_url": """https://static.wixstatic.com/media/f95dba_92a9978b127b40f693568167ce784038~mv2.png/v1/fill/w_1079,h_545,al_c,q_95,enc_auto/f95dba_92a9978b127b40f693568167ce784038~mv2.png""",
            "link": "https://diegsalaza.wixsite.com/miportalweb?lang=en"
        },                  
    ]

endorsements = {
    "img1": "https://user-images.githubusercontent.com/90204593/238169843-12872392-f2f1-40a6-a353-c06a2fa602c5.png",
    "img2": "https://user-images.githubusercontent.com/90204593/238171251-5f4c5597-84d4-4b4b-803c-afe74e739070.png",
    "img3": "https://user-images.githubusercontent.com/90204593/238171242-53f7ceb3-1a71-4726-a7f5-67721419fef8.png"
}

embed_rss= {
    'rss':"""<div style="overflow-y: scroll; height:500px; background-color:white;"> <div id="retainable-rss-embed" 
        data-rss="https://medium.com/feed/@vicky-note"
        data-maxcols="3" 
        data-layout="grid"
        data-poststyle="inline" 
        data-readmore="Read the rest" 
        data-buttonclass="btn btn-primary" 
        data-offset="0"></div></div> <script src="https://www.twilik.com/assets/retainable/rss-embed/retainable-rss-embed.js"></script>"""
}
