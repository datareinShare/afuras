import streamlit as st
import math
from typing import Dict, List, Tuple, Set
import re
import googlemaps
import pandas as pd


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    2点間の距離をキロメートルで計算（Haversine formula）
    """
    R = 6371  # 地球の半径（km）

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


class SchoolNameNormalizer:
    def __init__(self):
        """
        学校名を正規化するためのパターンリストを定義
        """
        self.removal_patterns = [
            r'市立',
            r'町立',
            r'村立',
            r'県立',
            r'私立',
            r'国立',
            r'学校法人',
            r'[都道府県]立',
            r'職員室',
            r'事務室',
            r'保健室',
            r'図書室',
            r'図書館',
            r'体育館',
            r'武道館',
            r'施設管理室',
            r'分室',
            r'校舎',
            r'第[一二三四五六七八九十]+',
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.removal_patterns]

    def normalize(self, name: str) -> str:
        """
        学校名を正規化して重複の識別を容易にする
        """
        normalized = name

        # 上記パターンに基づいて文字列を置換
        for pattern in self.compiled_patterns:
            normalized = pattern.sub('', normalized)

        # 空白の正規化
        normalized = ' '.join(normalized.split())

        return normalized.strip()


class SchoolClassifier:
    def __init__(self):
        """
        学校種別を判定するためのキーワードを定義
        """
        self.school_types = {
            "小学校": ["小学校"],
            "中学校": ["中学校"],
            "高校": ["高校", "高等学校"],
            "大学": ["大学", "大学院"]
        }

    def classify_school(self, name: str, place_types: List[str]) -> str:
        """
        学校名と場所タイプ情報から、学校種別（小・中・高・大学・その他）を判定する
        """
        # 大学の判定
        if "university" in place_types or any(kw in name for kw in self.school_types["大学"]):
            return "大学"

        # 小・中・高の判定
        for school_type, keywords in self.school_types.items():
            if any(kw in name for kw in keywords):
                return school_type

        return "その他"


class SchoolFinder:
    def __init__(self, api_key: str):
        """
        Google Maps APIクライアントや正規化・分類クラスを初期化
        """
        self.gmaps = googlemaps.Client(key=api_key)
        self.normalizer = SchoolNameNormalizer()
        self.classifier = SchoolClassifier()
        # 検索除外キーワード例（塾・予備校など）
        self.exclude_keywords = ["塾", "予備校", "サポート校", "学習塾"]

    def get_coordinates(self, station_name: str) -> Tuple[float, float]:
        """
        駅名をもとにジオコーディングして、緯度経度を取得する
        """
        try:
            geocode_result = self.gmaps.geocode(f"{station_name}駅", language='ja', region='jp')
            if not geocode_result:
                raise ValueError(f"位置情報が見つかりません: {station_name}")

            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        except Exception as e:
            st.error(f"エラー: 位置情報の取得に失敗しました - {str(e)}")
            return None, None

    def search_all_schools(self, center_location: Tuple[float, float], radius_km: float) -> List[Dict]:
        """
        指定された中心座標 (lat, lng) と検索半径 (km) をもとに
        Google Places API で学校を検索し、結果を返す。
        """
        schools = {}
        # 半径(m)に1.5倍をかけた値でAPIに問い合わせる
        radius_meters = int(radius_km * 1000 * 1.5)
        search_keywords = ["学校", "小学校", "中学校", "高校", "高等学校", "大学"]

        progress_bar = st.progress(0)
        total_keywords = len(search_keywords)
        
        for i, keyword in enumerate(search_keywords):
            try:
                places_result = self.gmaps.places_nearby(
                    location=center_location,
                    radius=radius_meters,
                    keyword=keyword,
                    language='ja'
                )

                if places_result.get('results'):
                    for place in places_result['results']:
                        # 除外キーワードのチェック
                        if any(exclude in place['name'] for exclude in self.exclude_keywords):
                            continue

                        # 距離を計算し、radius_km以内なら取り込む
                        school_lat = place['geometry']['location']['lat']
                        school_lng = place['geometry']['location']['lng']
                        distance_km = calculate_distance(center_location[0], center_location[1],
                                                         school_lat, school_lng)

                        if distance_km <= radius_km:
                            # 学校名を正規化
                            original_name = place['name']
                            normalized_name = self.normalizer.normalize(original_name)

                            if normalized_name:
                                # 学校種別を判定
                                school_type = self.classifier.classify_school(
                                    original_name,
                                    place.get('types', [])
                                )

                                # 重複チェック: 学校名+学校種別が同じ場合、より近いもののみ保持
                                key = f"{normalized_name}_{school_type}"
                                if key not in schools or schools[key]['distance'] > distance_km:
                                    schools[key] = {
                                        'normalized_name': normalized_name,
                                        'original_name': original_name,
                                        'distance': distance_km,
                                        'type': school_type,
                                        'location': (school_lat, school_lng)
                                    }

                progress_bar.progress((i + 1) / total_keywords)
                
            except Exception as e:
                st.warning(f"警告: キーワード「{keyword}」の検索中にエラーが発生しました - {str(e)}")
                continue

        progress_bar.empty()
        return list(schools.values())

    def organize_results(self, schools: List[Dict]) -> Dict[str, List[Dict]]:
        """
        学校一覧を学校種別ごとにまとめて返す
        """
        organized = {
            "小学校": [],
            "中学校": [],
            "高校": [],
            "大学": [],
            "その他": []
        }

        for school in schools:
            school_type = school['type']
            if school_type in organized:
                organized[school_type].append(school)

        # 各カテゴリ内のリストを距離でソート
        for category in organized:
            organized[category].sort(key=lambda x: x['distance'])

        return organized


def main():
    st.set_page_config(
        page_title="学校検索アプリ",
        page_icon="🏫",
        layout="wide"
    )
    
    st.title("🏫 駅周辺学校検索アプリ")
    st.markdown("---")
    
    # APIキーの取得（優先順位: secrets.toml > 環境変数 > 手動入力）
    api_key = None
    
    # 1. secrets.tomlから取得を試行
    try:
        api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
    except:
        pass
    
    # 2. 環境変数から取得を試行
    if not api_key:
        import os
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    # 3. 手動入力（開発時のみ）
    if not api_key:
        with st.sidebar:
            st.header("⚙️ 設定")
            st.warning("APIキーが設定されていません")
            api_key = st.text_input(
                "Google Maps API Key", 
                type="password",
                help="Google Maps APIキーを入力してください（開発時のみ）"
            )
            
            if not api_key:
                st.error("Google Maps APIキーが必要です")
                st.info("本番環境では secrets.toml または環境変数でAPIキーを設定してください")
                st.stop()
    
    # ステップ1: 駅名入力
    st.header("📍 ステップ1: 駅名を入力")
    station_name = st.text_input(
        "駅名", 
        placeholder="例: 新宿、東京、名古屋",
        help="検索したい駅の名前を入力してください"
    )
    
    if not station_name:
        st.info("👆 まずは駅名を入力してください")
        st.stop()
    
    # 駅の座標を取得
    finder = SchoolFinder(api_key)
    
    if 'station_coords' not in st.session_state or st.session_state.get('current_station') != station_name:
        with st.spinner("駅の位置情報を取得中..."):
            lat, lng = finder.get_coordinates(station_name)
            if lat is None or lng is None:
                st.stop()
            
            st.session_state.station_coords = (lat, lng)
            st.session_state.current_station = station_name
    
    lat, lng = st.session_state.station_coords
    st.success(f"✅ {station_name}駅の位置情報を取得しました")
    
    # ステップ2: 半径入力
    st.header("📏 ステップ2: 検索半径を設定")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        radius_km = st.slider(
            "検索半径 (km)", 
            min_value=0.5, 
            max_value=5.0, 
            value=2.0, 
            step=0.1,
            help="駅から何km圏内の学校を検索するかを設定してください"
        )
    
    with col2:
        st.metric("設定半径", f"{radius_km:.1f} km")
    
    # ステップ3: 検索実行
    st.header("🔍 ステップ3: 検索実行")
    
    if st.button("🏫 学校を検索する", type="primary", use_container_width=True):
        with st.spinner("学校を検索中..."):
            all_schools = finder.search_all_schools((lat, lng), radius_km)
            organized_results = finder.organize_results(all_schools)
            
            st.session_state.search_results = organized_results
            st.session_state.search_station = station_name
            st.session_state.search_radius = radius_km
    
    # 結果表示
    if 'search_results' in st.session_state:
        st.header(f"📊 検索結果: {st.session_state.search_station}駅から{st.session_state.search_radius:.1f}km圏内")
        
        results = st.session_state.search_results
        total_schools = sum(len(schools) for schools in results.values())
        
        if total_schools == 0:
            st.warning("該当する学校がありませんでした。検索半径を広げてみてください。")
        else:
            # 概要表示
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("小学校", len(results["小学校"]))
            with col2:
                st.metric("中学校", len(results["中学校"]))
            with col3:
                st.metric("高校", len(results["高校"]))
            with col4:
                st.metric("大学", len(results["大学"]))
            with col5:
                st.metric("その他", len(results["その他"]))
            
            st.markdown("---")
            
            # 詳細結果をタブで表示
            tabs = st.tabs(["🏫 全体", "🎒 小学校", "📚 中学校", "🎓 高校", "🏛️ 大学", "📋 その他"])
            
            # 全体タブ
            with tabs[0]:
                all_schools_list = []
                for category, schools in results.items():
                    for school in schools:
                        all_schools_list.append({
                            "学校名": school['original_name'],
                            "種別": school['type'],
                            "距離 (km)": f"{school['distance']:.1f}"
                        })
                
                if all_schools_list:
                    df = pd.DataFrame(all_schools_list)
                    st.dataframe(df, use_container_width=True)
            
            # 各カテゴリタブ
            categories = ["小学校", "中学校", "高校", "大学", "その他"]
            for i, category in enumerate(categories):
                with tabs[i + 1]:
                    schools = results[category]
                    if schools:
                        school_data = []
                        for school in schools:
                            school_data.append({
                                "学校名": school['original_name'],
                                "距離 (km)": f"{school['distance']:.1f}"
                            })
                        
                        df = pd.DataFrame(school_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # 地図表示用のデータを準備
                        map_data = pd.DataFrame([
                            {
                                "lat": school['location'][0],
                                "lon": school['location'][1],
                                "school_name": school['original_name']
                            } for school in schools
                        ])
                        
                        if not map_data.empty:
                            st.subheader(f"{category}の位置")
                            st.map(map_data)
                    else:
                        st.info(f"{category}は見つかりませんでした。")


if __name__ == "__main__":
    main()