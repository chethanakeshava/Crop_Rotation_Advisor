"""
Crop Rotation Model
This model predicts the next two seasons' crops based on:
1. Initial crop recommendation from ML model
2. Crop family rotation rules (different families to disrupt pest cycles)
3. Season sequence (Kharif -> Rabi -> Zaid -> Kharif)
4. Soil health rotational rules (legume -> cereal -> vegetable pattern)
5. Custom rotation_rules for popular crop sequences
"""

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CropRotationModel:
    def __init__(
        self,
        model_pkl_path=r"C:\Users\cheth\OneDrive\Documents\Desktop\cursor_crop_rotation\crop_recommendation_model01.pkl",
        crop_family_csv=r"C:\Users\cheth\OneDrive\Documents\Desktop\cursor_crop_rotation\agricultural_data_with_crop_family.csv"
    ):
        """
        Initialize the Crop Rotation Model

        Args:
            model_pkl_path: Path to the saved ML model pkl file
            crop_family_csv: Path to CSV file with crop family information
        """
        # Load the ML model and preprocessors
        print("📦 Loading ML model and preprocessors...")
        with open(model_pkl_path, 'rb') as f:
            model_data = pickle.load(f)

        # From your training script
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']              # crop encoder (unchanged)
        self.label_encoder_season = model_data['label_encoder_season']  # season encoder (unchanged)
        self.feature_names = model_data['feature_names']              # same order as training
        self.numeric_feature_names = model_data['numeric_feature_names']

        # Load crop family data
        print("📊 Loading crop family data...")
        self.crop_family_df = pd.read_csv(crop_family_csv)
        self.crop_family_df = self.crop_family_df.drop(columns=['Unnamed: 0'], errors='ignore')

        # Create crop to family mapping
        self.crop_to_family = dict(zip(
            self.crop_family_df['Crop'].str.lower(),
            self.crop_family_df['Crop_Family']
        ))

        # Create family to crops mapping
        self.family_to_crops = {}
        for _, row in self.crop_family_df.iterrows():
            family = row['Crop_Family']
            crop = row['Crop'].lower()
            if family not in self.family_to_crops:
                self.family_to_crops[family] = set()
            self.family_to_crops[family].add(crop)

        # Create crop to season mapping
        self.crop_to_seasons = {}
        for _, row in self.crop_family_df.iterrows():
            crop = row['Crop'].lower()
            season = row['Season']
            if crop not in self.crop_to_seasons:
                self.crop_to_seasons[crop] = set()
            self.crop_to_seasons[crop].add(season)

        # Define season sequence
        self.season_sequence = {
            'Kharif': 'Rabi',
            'Rabi': 'Zaid',
            'Zaid': 'Kharif'
        }

        # Seasonal rotation rules (informational)
        self.seasonal_rotation_rules = {
            'kharif_season': {
                'period': 'June/July to September/October (Monsoon season)',
                'climate': 'Hot and humid conditions with substantial rainfall',
                'crops_from_map': ['rice', 'maize', 'jowar', 'soyabean', 'moong', 'blackgram',
                                   'bittergourd', 'bottlegourd', 'pumpkin', 'ladyfinger', 'cucumber',
                                   'sweetpotato', 'sunflower', 'horsegram'],
            },
            'rabi_season': {
                'period': 'October/November to March/April (Winter season)',
                'climate': 'Cooler temperatures; less water requirement',
                'crops_from_map': ['barley', 'wheat', 'rapeseed', 'potato', 'onion', 'garlic',
                                   'cabbage', 'cauliflower', 'radish', 'coriander', 'ragi', 'horsegram'],
            },
            'zaid_season': {
                'period': 'March to June (Short summer between Rabi and Kharif)',
                'climate': 'Warm, dry weather',
                'crops_from_map': ['cucumber', 'bittergourd', 'bottlegourd', 'pumpkin', 'tomato',
                                   'brinjal', 'ladyfinger'],
            }
        }

        # 🔁 Your explicit rotation rules (boosted in scoring)
        self.rotation_rules = {
            "Kharif_to_Rabi": {
                "rice": ["barley", "wheat", "cabbage", "radish"],
                "soyabean": ["wheat", "barley", "coriander"],
                "jowar": ["sweetpotato", "potato", "onion"],
                "maize": ["garlic", "cabbage", "radish"],
                "ragi": ["wheat", "rapeseed", "coriander"],
                "horsegram": ["potato", "sweetpotato", "onion"],
                "blackgram": ["wheat", "barley", "cabbage"]
            },

            "Rabi_to_Zaid": {
                "barley": ["sunflower", "moong", "cucumber"],
                "sweetpotato": ["cucumber", "bittergourd", "pumpkin"],
                "potato": ["bottlegourd", "cucumber", "tomato"],
                "onion": ["moong", "sunflower", "pumpkin"],
                "cabbage": ["sunflower", "bittergourd", "cucumber"],
                "rapeseed": ["moong", "tomato", "cucumber"],
                "wheat": ["sunflower", "moong", "bottlegourd"],
                "coriander": ["cucumber", "pumpkin", "tomato"],
                "garlic": ["moong", "sunflower", "bittergourd"],
                "cauliflower": ["cucumber", "moong", "tomato"],
                "radish": ["sunflower", "pumpkin", "bottlegourd"]
            },

            "Zaid_to_Kharif": {
                "sunflower": ["rice", "maize", "ragi"],
                "moong": ["rice", "jowar", "maize"],
                "tomato": ["soyabean", "ragi", "horsegram"],
                "brinjal": ["soyabean", "jowar", "maize"],
                "bittergourd": ["rice", "jowar", "horsegram"],
                "ladyfinger": ["soyabean", "maize", "ragi"],
                "bottlegourd": ["rice", "jowar", "horsegram"],
                "pumpkin": ["soyabean", "jowar", "maize"],
                "cucumber": ["rice", "soyabean", "maize"]
            }
        }

        # Define crop families (for legume/cereal/vegetable logic)
        self.fabaceae_crops = ['soyabean', 'moong', 'blackgram', 'horsegram']  # Legumes - nitrogen fixers
        self.poaceae_crops = ['rice', 'maize', 'jowar', 'wheat', 'barley', 'ragi']  # Cereals - heavy feeders
        self.solanaceae_crops = ['potato', 'tomato', 'brinjal']
        self.brassicaceae_crops = ['cabbage', 'cauliflower', 'rapeseed', 'radish']
        self.cucurbitaceae_crops = ['cucumber', 'bittergourd', 'bottlegourd', 'pumpkin']

        # Root depth classification (simplified)
        self.deep_rooted_crops = ['horsegram', 'sunflower', 'jowar', 'maize']
        self.shallow_rooted_crops = ['potato', 'onion', 'garlic', 'radish', 'coriander']

        print("✅ Crop Rotation Model initialized successfully!")

    # ---------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------
    def get_next_season(self, current_season):
        """Get the next season in the rotation cycle"""
        return self.season_sequence.get(current_season, 'Kharif')

    def get_season_after_next(self, current_season):
        """Get the season after next in the rotation cycle"""
        next_season = self.get_next_season(current_season)
        return self.get_next_season(next_season)

    # ---------------------------------------------------------
    # ML prediction helpers
    # ---------------------------------------------------------
    def predict_crop(self, N, P, K, temperature, pH, rainfall, season):
        """
        Predict crop based on soil and weather conditions

        Returns:
            Recommended crop name (lowercase)
        """
        sample_input = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'pH': pH,
            'rainfall': rainfall,
            'Season_Label': self.label_encoder_season.transform([season])[0]
        }

        # DataFrame with same feature order as training
        sample_df = pd.DataFrame([sample_input])
        sample_df = sample_df.reindex(columns=self.numeric_feature_names, fill_value=0)

        # Scale input and predict
        sample_scaled = self.scaler.transform(sample_df)
        predicted_label = self.model.predict(sample_scaled)[0]
        predicted_crop = self.label_encoder.inverse_transform([predicted_label])[0]

        return predicted_crop.lower()

    def predict_crop_with_constraints(
        self, N, P, K, temperature, pH, rainfall, season, allowed_crops=None
    ):
        """
        Predict crop based on conditions, but only from allowed crops list.

        allowed_crops: list of crop names (lowercase).
        """
        if allowed_crops is None or len(allowed_crops) == 0:
            return self.predict_crop(N, P, K, temperature, pH, rainfall, season).title()

        all_crops = list(self.label_encoder.classes_)
        allowed_crops_lower = [c.lower() for c in allowed_crops]
        valid_crops = [c for c in all_crops if c.lower() in allowed_crops_lower]

        if not valid_crops:
            return None

        sample_input = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'pH': pH,
            'rainfall': rainfall,
            'Season_Label': self.label_encoder_season.transform([season])[0]
        }

        sample_df = pd.DataFrame([sample_input])
        sample_df = sample_df.reindex(columns=self.numeric_feature_names, fill_value=0)
        sample_scaled = self.scaler.transform(sample_df)

        prediction_proba = self.model.predict_proba(sample_scaled)[0]

        crop_proba_dict = {}
        for i, crop in enumerate(all_crops):
            if crop.lower() in allowed_crops_lower:
                crop_proba_dict[crop] = prediction_proba[i]

        if not crop_proba_dict:
            return None

        best_crop = max(crop_proba_dict, key=crop_proba_dict.get)
        return best_crop.title()

    # ---------------------------------------------------------
    # Agronomic helpers
    # ---------------------------------------------------------
    def get_crop_family(self, crop_name):
        return self.crop_to_family.get(crop_name.lower(), None)

    def get_crops_in_family(self, family_name, exclude_crop=None):
        crops = self.family_to_crops.get(family_name, set())
        if exclude_crop:
            crops = crops - {exclude_crop.lower()}
        return list(crops)

    def get_crops_for_season(self, crops_list, season):
        suitable = []
        for crop in crops_list:
            crop_seasons = self.crop_to_seasons.get(crop.lower(), set())
            if season in crop_seasons:
                suitable.append(crop)
        return suitable

    def is_legume(self, crop_name):
        return crop_name.lower() in self.fabaceae_crops

    def is_cereal(self, crop_name):
        return crop_name.lower() in self.poaceae_crops

    def is_vegetable(self, crop_name):
        c = crop_name.lower()
        return (c not in self.fabaceae_crops and c not in self.poaceae_crops)

    def is_deep_rooted(self, crop_name):
        return crop_name.lower() in self.deep_rooted_crops

    def is_shallow_rooted(self, crop_name):
        return crop_name.lower() in self.shallow_rooted_crops

    def get_crops_from_different_family(self, exclude_family, exclude_crop=None):
        all_crops = []
        for crop, family in self.crop_to_family.items():
            if family != exclude_family:
                if exclude_crop is None or crop != exclude_crop.lower():
                    all_crops.append(crop)
        return all_crops

    def is_rotation_compatible(self, previous_crop, candidate_crop):
        prev_family = self.get_crop_family(previous_crop)
        cand_family = self.get_crop_family(candidate_crop)

        if prev_family == cand_family:
            return False, "Same family - should rotate to different family"

        if ((prev_family == 'Solanaceae' and cand_family == 'Brassicaceae') or
            (prev_family == 'Brassicaceae' and cand_family == 'Solanaceae')):
            return False, "Solanaceae and Brassicaceae should not follow each other"

        return True, "Compatible"

    def score_crop_for_rotation(self, previous_crop, candidate_crop, season):
        score = 0
        prev_family = self.get_crop_family(previous_crop)
        cand_family = self.get_crop_family(candidate_crop)

        # Base: different family
        if prev_family != cand_family:
            score += 10

        # Legume -> Cereal (very good)
        if self.is_legume(previous_crop) and self.is_cereal(candidate_crop):
            score += 20
            if season == 'Rabi':
                score += 10

        # Cereal -> Vegetable
        if self.is_cereal(previous_crop) and self.is_vegetable(candidate_crop):
            score += 15

        # Deep-rooted -> shallow-rooted
        if self.is_deep_rooted(previous_crop) and self.is_shallow_rooted(candidate_crop):
            score += 10

        # Penalties
        if prev_family == cand_family:
            score -= 50

        if ((prev_family == 'Solanaceae' and cand_family == 'Brassicaceae') or
            (prev_family == 'Brassicaceae' and cand_family == 'Solanaceae')):
            score -= 50

        return score

    def _rotation_bonus_from_table(self, from_season, to_season, from_crop, candidate_crop):
        """
        Extra +score if (from_crop -> candidate_crop) is present in rotation_rules table.
        """
        key = f"{from_season}_to_{to_season}"
        mapping = self.rotation_rules.get(key, {})
        allowed_list = mapping.get(from_crop.lower(), [])
        if candidate_crop.lower() in [c.lower() for c in allowed_list]:
            return 15  # strong bonus
        return 0

    # ---------------------------------------------------------
    # Main rotation prediction
    # ---------------------------------------------------------
    def predict_rotation(self, N, P, K, temperature, pH, rainfall, input_season):
        print(f"\n🌾 Predicting crop rotation for Season: {input_season}")
        print("=" * 60)

        # Season 1: ML recommended
        print(f"\n📊 Step 1: Predicting crop based on conditions...")
        recommended_crop = self.predict_crop(N, P, K, temperature, pH, rainfall, input_season)

        # 🔥 Hard restriction: Avoid wheat in Zaid
        if input_season == "Zaid" and recommended_crop.lower() == "wheat":
            print("⚠ Wheat cannot be grown in Zaid — applying correction rule...")
            non_poaceae = self.get_crops_from_different_family("Poaceae", exclude_crop="wheat")
            alternatives = self.get_crops_for_season(non_poaceae, "Zaid")
            if alternatives:
                recommended_crop = alternatives[0]
                print(f"➡ Replacing Wheat with {recommended_crop.title()} for Zaid")
            else:
                print("⚠ No suitable non-wheat alternative found; keeping model prediction.")

        print(f"✅ Recommended Crop: {recommended_crop.title()}")

        crop_family = self.get_crop_family(recommended_crop)
        if not crop_family:
            print(f"⚠️  Warning: Crop family not found for {recommended_crop}")
            return None

        print(f"✅ Crop Family: {crop_family}")

        different_family_crops = self.get_crops_from_different_family(crop_family, exclude_crop=recommended_crop)
        print(f"✅ Crops from different families (excluding {crop_family}): {len(different_family_crops)} crops")

        if not different_family_crops:
            print(f"⚠️  Warning: No crops found from different families")
            different_family_crops = list(self.crop_to_family.keys())
            different_family_crops = [c for c in different_family_crops if c != recommended_crop]

        # ---------------- Season 2 ----------------
        next_season = self.get_next_season(input_season)
        print(f"\n📅 Step 2: Predicting crop for next season ({next_season})...")
        next_season_crops = self.get_crops_for_season(different_family_crops, next_season)
        print(f"✅ Crops suitable for {next_season} from different families: {len(next_season_crops)} crops")

        if next_season_crops:
            compatible_crops = []
            for crop in next_season_crops:
                is_compat, _ = self.is_rotation_compatible(recommended_crop, crop)
                if is_compat:
                    compatible_crops.append(crop)

            if compatible_crops:
                crop_scores = {}
                for crop in compatible_crops:
                    base_score = self.score_crop_for_rotation(recommended_crop, crop, next_season)
                    # 🔁 Extra bonus if (recommended → crop) in your rotation_rules table
                    bonus = self._rotation_bonus_from_table(input_season, next_season, recommended_crop, crop)
                    total_score = base_score + bonus

                    # ML constraint check
                    ml_pred = self.predict_crop_with_constraints(
                        N, P, K, temperature, pH, rainfall, next_season,
                        allowed_crops=[crop]
                    )
                    if ml_pred:
                        crop_scores[crop] = total_score

                if crop_scores:
                    sorted_crops = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
                    top_candidates = [c[0] for c in sorted_crops[:3]]
                    next_season_crop = self.predict_crop_with_constraints(
                        N, P, K, temperature, pH, rainfall, next_season,
                        allowed_crops=top_candidates
                    )
                    if not next_season_crop:
                        next_season_crop = top_candidates[0].title()
                    print(f"✅ Next Season Crop (ML + Rotation Rules): {next_season_crop}")
                else:
                    next_season_crop = self.predict_crop_with_constraints(
                        N, P, K, temperature, pH, rainfall, next_season,
                        allowed_crops=compatible_crops
                    )
                    if not next_season_crop:
                        next_season_crop = compatible_crops[0].title()
                    print(f"✅ Next Season Crop: {next_season_crop}")
            else:
                print(f"⚠️  No rotation-compatible crops found, using ML model...")
                next_season_crop = self.predict_crop_with_constraints(
                    N, P, K, temperature, pH, rainfall, next_season,
                    allowed_crops=next_season_crops
                )
                if not next_season_crop:
                    next_season_crop = next_season_crops[0].title()
                print(f"✅ Next Season Crop: {next_season_crop}")
        else:
            print(f"⚠️  No crops from different families for {next_season}, searching all crops...")
            all_crops = list(self.crop_to_seasons.keys())
            fallback_crops = self.get_crops_for_season(all_crops, next_season)
            if fallback_crops:
                next_season_crop = self.predict_crop_with_constraints(
                    N, P, K, temperature, pH, rainfall, next_season,
                    allowed_crops=fallback_crops
                )
                if not next_season_crop:
                    next_season_crop = fallback_crops[0].title()
                print(f"✅ Next Season Crop: {next_season_crop}")
            else:
                next_season_crop = "No suitable crop found"
                print(f"❌ No suitable crop found for {next_season}")

        # ---------------- Season 3 ----------------
        season_after_next = self.get_season_after_next(input_season)
        print(f"\n📅 Step 3: Predicting crop for season after next ({season_after_next})...")

        if next_season_crop and next_season_crop != "No suitable crop found":
            next_crop_family = self.get_crop_family(next_season_crop.lower())
            if next_crop_family:
                print(f"   Using ML model with rotation rules (different family from {next_crop_family})...")
                different_family_crops_2 = self.get_crops_from_different_family(
                    next_crop_family, exclude_crop=next_season_crop.lower()
                )
                season_after_next_crops = self.get_crops_for_season(different_family_crops_2, season_after_next)
                print(f"✅ Crops suitable for {season_after_next} from different families: {len(season_after_next_crops)} crops")

                if season_after_next_crops:
                    compatible_crops_2 = []
                    for crop in season_after_next_crops:
                        is_compat, _ = self.is_rotation_compatible(next_season_crop.lower(), crop)
                        if is_compat:
                            compatible_crops_2.append(crop)

                    if compatible_crops_2:
                        crop_scores_2 = {}
                        for crop in compatible_crops_2:
                            base_score = self.score_crop_for_rotation(next_season_crop.lower(), crop, season_after_next)
                            # 🔁 Bonus from your table (Season2 → Season3)
                            bonus = self._rotation_bonus_from_table(next_season, season_after_next, next_season_crop, crop)
                            crop_scores_2[crop] = base_score + bonus

                        if crop_scores_2:
                            sorted_crops_2 = sorted(crop_scores_2.items(), key=lambda x: x[1], reverse=True)
                            top_candidates_2 = [c[0] for c in sorted_crops_2[:3]]
                            season_after_next_crop = self.predict_crop_with_constraints(
                                N, P, K, temperature, pH, rainfall, season_after_next,
                                allowed_crops=top_candidates_2
                            )
                            if not season_after_next_crop:
                                season_after_next_crop = top_candidates_2[0].title()
                            print(f"✅ Season After Next Crop: {season_after_next_crop}")
                        else:
                            season_after_next_crop = self.predict_crop_with_constraints(
                                N, P, K, temperature, pH, rainfall, season_after_next,
                                allowed_crops=compatible_crops_2
                            )
                            if not season_after_next_crop:
                                season_after_next_crop = compatible_crops_2[0].title()
                            print(f"✅ Season After Next Crop: {season_after_next_crop}")
                    else:
                        season_after_next_crop = self.predict_crop_with_constraints(
                            N, P, K, temperature, pH, rainfall, season_after_next,
                            allowed_crops=season_after_next_crops
                        )
                        if not season_after_next_crop:
                            season_after_next_crop = season_after_next_crops[0].title()
                        print(f"✅ Season After Next Crop: {season_after_next_crop}")
                else:
                    print(f"⚠️  No crops from different families for {season_after_next}, searching all crops...")
                    all_crops = list(self.crop_to_seasons.keys())
                    fallback_crops = self.get_crops_for_season(all_crops, season_after_next)
                    if fallback_crops:
                        season_after_next_crop = self.predict_crop_with_constraints(
                            N, P, K, temperature, pH, rainfall, season_after_next,
                            allowed_crops=fallback_crops
                        )
                        if not season_after_next_crop:
                            season_after_next_crop = fallback_crops[0].title()
                        print(f"✅ Season After Next Crop: {season_after_next_crop}")
                    else:
                        season_after_next_crop = "No suitable crop found"
                        print(f"❌ No suitable crop found for {season_after_next}")
            else:
                print(f"⚠️  Family not found for {next_season_crop}, searching all crops...")
                all_crops = list(self.crop_to_seasons.keys())
                fallback_crops = self.get_crops_for_season(all_crops, season_after_next)
                if fallback_crops:
                    season_after_next_crop = self.predict_crop_with_constraints(
                        N, P, K, temperature, pH, rainfall, season_after_next,
                        allowed_crops=fallback_crops
                    )
                    if not season_after_next_crop:
                        season_after_next_crop = fallback_crops[0].title()
                    print(f"✅ Season After Next Crop: {season_after_next_crop}")
                else:
                    season_after_next_crop = "No suitable crop found"
                    print(f"❌ No suitable crop found for {season_after_next}")
        else:
            print(f"⚠️  Next season crop not found, searching all crops for {season_after_next}...")
            all_crops = list(self.crop_to_seasons.keys())
            fallback_crops = self.get_crops_for_season(all_crops, season_after_next)
            if fallback_crops:
                season_after_next_crop = self.predict_crop_with_constraints(
                    N, P, K, temperature, pH, rainfall, season_after_next,
                    allowed_crops=fallback_crops
                )
                if not season_after_next_crop:
                    season_after_next_crop = fallback_crops[0].title()
                print(f"✅ Season After Next Crop: {season_after_next_crop}")
            else:
                season_after_next_crop = "No suitable crop found"
                print(f"❌ No suitable crop found for {season_after_next}")

        # 🔥 Also enforce "no wheat in Zaid" for Season 3 as well
        if season_after_next == "Zaid" and season_after_next_crop != "No suitable crop found":
            if season_after_next_crop.lower() == "wheat":
                print("⚠ Wheat cannot be grown in Zaid (Season 3) — applying correction rule...")
                non_poaceae = self.get_crops_from_different_family("Poaceae", exclude_crop="wheat")
                alternatives_zaid3 = self.get_crops_for_season(non_poaceae, "Zaid")
                if alternatives_zaid3:
                    season_after_next_crop = alternatives_zaid3[0].title()
                    print(f"➡ Replacing Wheat with {season_after_next_crop} for Zaid (Season 3)")

        # Families
        next_crop_family = self.get_crop_family(next_season_crop.lower()) if next_season_crop != "No suitable crop found" else "Unknown"
        after_next_crop_family = self.get_crop_family(season_after_next_crop.lower()) if season_after_next_crop != "No suitable crop found" else "Unknown"

        result = {
            'recommended_crop': recommended_crop.title(),
            'crop_family': crop_family,
            'input_season': input_season,
            'next_season': next_season,
            'next_season_crop': next_season_crop,
            'next_season_crop_family': next_crop_family,
            'season_after_next': season_after_next,
            'season_after_next_crop': season_after_next_crop,
            'season_after_next_crop_family': after_next_crop_family
        }

        # Pretty print
        print("\n" + "=" * 60)
        print("\n🌱 SEASONAL CROP ROTATION PLAN\n")
        print("=" * 60)

        print(f"\nSeason 1: {input_season} (Current Season)")
        print(f"▶ Recommended Crop:       {result['recommended_crop']} ({crop_family})")

        if next_season_crop != "No suitable crop found":
            print(f"\nSeason 2: {next_season} (Next Season)")
            print(f"▶ Rotation Crop:          {result['next_season_crop']} ({next_crop_family})")
        else:
            print(f"\nSeason 2: {next_season} (Next Season)")
            print(f"▶ Rotation Crop:          {next_season_crop}")

        if season_after_next_crop != "No suitable crop found":
            print(f"\nSeason 3: {season_after_next} (Season After Next)")
            print(f"▶ Rotation Crop:          {result['season_after_next_crop']} ({after_next_crop_family})")
        else:
            print(f"\nSeason 3: {season_after_next} (Season After Next)")
            print(f"▶ Rotation Crop:          {season_after_next_crop}")

        print("\n" + "-" * 60)
        print("\n🟩 Benefits of this rotation:\n")

        overall_benefits = []

        if (self.is_legume(recommended_crop) or
            (next_season_crop != "No suitable crop found" and self.is_legume(next_season_crop)) or
            (season_after_next_crop != "No suitable crop found" and self.is_legume(season_after_next_crop))):
            overall_benefits.append(" • Sustainable nutrient cycling")

        if (crop_family != next_crop_family and
            (season_after_next_crop == "No suitable crop found" or next_crop_family != after_next_crop_family)):
            overall_benefits.append(" • Lower disease and pest risk")

        if (self.is_legume(recommended_crop) or
            (next_season_crop != "No suitable crop found" and self.is_legume(next_season_crop))):
            overall_benefits.append(" • Improved soil fertility")

        if (input_season in ['Kharif', 'Rabi', 'Zaid'] and
            next_season in ['Kharif', 'Rabi', 'Zaid'] and
            season_after_next in ['Kharif', 'Rabi', 'Zaid']):
            overall_benefits.append(" • Efficient seasonal land use")

        if not overall_benefits:
            overall_benefits = [
                " • Sustainable nutrient cycling",
                " • Lower disease and pest risk",
                " • Improved soil fertility",
                " • Efficient seasonal land use"
            ]

        for benefit in overall_benefits:
            print(benefit)

        print("\n" + "=" * 60)

        return result


if __name__ == "__main__":
    model = CropRotationModel()

    N = float(input("Enter N: ") or 80)
    P = float(input("Enter P: ") or 60)
    K = float(input("Enter K: ") or 20)
    pH = float(input("Enter pH: ") or 5.5)
    rainfall = float(input("Enter rainfall: ") or 300)
    temperature = float(input("Enter temperature: ") or 29)
    season = input("Enter season (Kharif/Rabi/Zaid): ") or "Rabi"

    model.predict_rotation(N, P, K, temperature, pH, rainfall, season)
