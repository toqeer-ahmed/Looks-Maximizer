import os
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import base64

class ModelManager:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Go up one level from 'backend' to root, then into 'ml_pipeline'
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, "ml_pipeline")
        else:
            self.model_dir = model_dir
            
        print(f"Looking for models in: {self.model_dir}")
        self.models = {}
        self.load_models()

    def load_models(self):
        # Helper to safely load a model
        def load_safe(name, filename):
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                try:
                    # Disable external data loading checks if causing issues, or ensure paths are correct
                    sess_options = ort.SessionOptions()
                    # sess_options.log_severity_level = 3
                    self.models[name] = ort.InferenceSession(path, sess_options)
                    print(f"Loaded {name} model from {path}")
                except Exception as e:
                    print(f"Error loading {name} model: {e}")
            else:
                print(f"Warning: {name} model not found at {path}")

        # 1. CelebA Model
        load_safe('celeba', "celeba_multitask.onnx")

        # 2. HF Model
        load_safe('hf', "hf_model.onnx")

        # 3. SCUT Model
        load_safe('scut', "scut_model.onnx")

    def preprocess(self, image, size=(224, 224)):
        # Resize
        img = cv2.resize(image, size)
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize (Mean/Std from ImageNet)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        # CHW format
        img = img.transpose(2, 0, 1)
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
import os
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import base64

class ModelManager:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Go up one level from 'backend' to root, then into 'ml_pipeline'
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, "ml_pipeline")
        else:
            self.model_dir = model_dir
            
        print(f"Looking for models in: {self.model_dir}")
        self.models = {}
        self.load_models()

    def load_models(self):
        # Helper to safely load a model
        def load_safe(name, filename):
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                try:
                    # Disable external data loading checks if causing issues, or ensure paths are correct
                    sess_options = ort.SessionOptions()
                    # sess_options.log_severity_level = 3
                    self.models[name] = ort.InferenceSession(path, sess_options)
                    print(f"Loaded {name} model from {path}")
                except Exception as e:
                    print(f"Error loading {name} model: {e}")
            else:
                print(f"Warning: {name} model not found at {path}")

        # 1. CelebA Model
        load_safe('celeba', "celeba_multitask.onnx")

        # 2. HF Model
        load_safe('hf', "hf_model.onnx")

        # 3. SCUT Model
        load_safe('scut', "scut_model.onnx")

    def preprocess(self, image, size=(224, 224)):
        # Resize
        img = cv2.resize(image, size)
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize (Mean/Std from ImageNet)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        # CHW format
        img = img.transpose(2, 0, 1)
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def preprocess_scut(self, image, size=(256, 256)):
        # SCUT model was trained on 256x256
        return self.preprocess(image, size)

    def predict(self, image_source, user_details={}):
        # Load Image
        image = None
        try:
            if image_source.startswith("http"):
                resp = urllib.request.urlopen(image_source)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            elif image_source.startswith("data:image"):
                # Handle Base64
                encoded_data = image_source.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(image_source)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

        if image is None:
            print("Failed to decode image")
            return None

        results = {}

        # --- Run CelebA Inference ---
        if 'celeba' in self.models:
            try:
                input_tensor = self.preprocess(image, size=(224, 224))
                input_name = self.models['celeba'].get_inputs()[0].name
                outputs = self.models['celeba'].run(None, {input_name: input_tensor})
                
                def sigmoid(x): return 1 / (1 + np.exp(-x))
                
                hair_logits = outputs[0][0]
                beard_logits = outputs[1][0]
                glasses_prob = sigmoid(outputs[2][0][0])
                attractive_prob = sigmoid(outputs[3][0][0])
                gender_prob = sigmoid(outputs[4][0][0])
                smiling_prob = sigmoid(outputs[5][0][0])

                hair_labels = ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Straight_Hair", "Wavy_Hair"]
                beard_labels = ["No_Beard", "Mustache", "Goatee", "Sideburns"]

                results['hair'] = hair_labels[np.argmax(hair_logits)]
                results['beard'] = beard_labels[np.argmax(beard_logits)] if np.max(sigmoid(beard_logits)) > 0.5 else "Clean Shaven"
                results['glasses'] = bool(glasses_prob > 0.5)
                results['attractive_score_celeba'] = float(attractive_prob)
                results['gender_celeba'] = "Male" if gender_prob > 0.5 else "Female"
                results['smiling'] = bool(smiling_prob > 0.5)
            except Exception as e:
                print(f"CelebA Inference Error: {e}")

        # --- Run HF Inference ---
        if 'hf' in self.models:
            try:
                input_tensor = self.preprocess(image, size=(224, 224))
                input_name = self.models['hf'].get_inputs()[0].name
                outputs = self.models['hf'].run(None, {input_name: input_tensor})
                
                age_logits = outputs[0][0]
                gender_logits = outputs[1][0]
                race_logits = outputs[2][0]
                
                age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
                race_labels = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]

                results['age_group'] = age_labels[np.argmax(age_logits)]
                results['gender_hf'] = "Male" if gender_logits[0] > 0 else "Female"
                results['race'] = race_labels[np.argmax(race_logits)]
            except Exception as e:
                print(f"HF Inference Error: {e}")

        # --- Run SCUT Inference ---
        if 'scut' in self.models:
            try:
                input_tensor = self.preprocess_scut(image, size=(256, 256))
                input_name = self.models['scut'].get_inputs()[0].name
                outputs = self.models['scut'].run(None, {input_name: input_tensor})
                
                beauty_score = float(outputs[0])
                results['beauty_score'] = round(beauty_score, 2)
                results['lookScore'] = int(min(max((beauty_score / 5.0) * 100, 0), 100))
            except Exception as e:
                print(f"SCUT Inference Error: {e}")
        
        # Default Look Score if SCUT failed
        if 'lookScore' not in results:
            results['lookScore'] = int(results.get('attractive_score_celeba', 0.5) * 100)

        # --- Generate Recommendations ---
        # Merge inferred attributes with user provided details
        combined_attributes = {**results, **user_details}
        results['recommendations'] = self.generate_recommendations(combined_attributes)
        results['faceShape'] = "Oval" # Placeholder

        return results

    def generate_recommendations(self, attributes):
        recs = {
            "hairstyles": [],
            "beardStyles": [],
            "clothingStyle": []
        }
        
        # 1. Gender & Age Context
        gender = attributes.get('gender', attributes.get('gender_hf', attributes.get('gender_celeba', 'Male')))
        age_group = attributes.get('age_group', '20-29')
        user_age = int(attributes.get('age', 25)) if attributes.get('age') else 25
        
        # 2. Physical Attributes
        height_str = attributes.get('height', '') # e.g. "5'10"
        skin_tone = attributes.get('skinTone', 'Medium')
        face_shape = attributes.get('faceShape', 'Oval')
        
        # --- Logic ---
        
        if gender == 'Male':
            # Hair
            if face_shape == 'Square':
                recs['hairstyles'] = ["Classic Undercut (Balances strong jaw)", "Quiff (Adds height)", "Side Part"]
            elif face_shape == 'Oval':
                recs['hairstyles'] = ["Pompadour", "Buzz Cut (Show off balanced features)", "Faux Hawk"]
            else:
                recs['hairstyles'] = ["Textured Crop", "Messy Fringe", "Crew Cut"]
                
            # Beard
            if user_age < 20:
                recs['beardStyles'] = ["Clean Shaven (Youthful look)", "Light Stubble"]
            else:
                recs['beardStyles'] = ["Short Boxed Beard (Professional)", "Heavy Stubble (Rugged)"]
                
            # Clothing (Height & Skin Tone)
            if 'Short' in height_str or (height_str and height_str < "5'7"): # Simple string check for now
                recs['clothingStyle'].append("Vertical Stripes (Elongate torso)")
                recs['clothingStyle'].append("Monochromatic Outfits (Unbroken line)")
            elif 'Tall' in height_str or (height_str and height_str > "6'0"):
                recs['clothingStyle'].append("Contrasting Separates (Break up height)")
                recs['clothingStyle'].append("Cuffed Pants")
            else:
                recs['clothingStyle'].append("Fitted T-shirts")
                recs['clothingStyle'].append("Slim-fit Jeans")

            if skin_tone in ['Fair', 'Light']:
                recs['clothingStyle'].append("Colors: Navy, Charcoal, Burgundy (Contrast)")
            elif skin_tone in ['Medium', 'Olive']:
                recs['clothingStyle'].append("Colors: Beige, Olive, Cream (Earth tones)")
            elif skin_tone in ['Dark', 'Brown']:
                recs['clothingStyle'].append("Colors: White, Pastel Pink, Light Blue (Pop against skin)")

        else: # Female
            # Hair
            recs['hairstyles'] = ["Long Layers (Framing)", "Bob Cut (Chic)", "Beach Waves (Volume)"]
            
            # Clothing
            if skin_tone in ['Fair', 'Light']:
                recs['clothingStyle'].append("Jewel Tones (Emerald, Royal Blue)")
            elif skin_tone in ['Medium', 'Olive']:
                recs['clothingStyle'].append("Metallics & Earth Tones")
            else:
                recs['clothingStyle'].append("Bright Colors (Yellow, Cobalt Blue)")
                
            if 'Short' in height_str:
                recs['clothingStyle'].append("High-waisted Jeans (Leg lengthening)")
            
        return recs

# Singleton instance
model_manager = ModelManager()
