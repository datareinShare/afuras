import streamlit as st
import math
from typing import Dict, List, Tuple, Set
import re
import googlemaps
import pandas as pd
import requests
import json


def send_analysis_data_to_gas(station_name: str, analysis_results: Dict = None, school_details: Dict = None) -> bool:
    """
    駅名と分析結果をGoogle Apps ScriptのURLにPOSTリクエストで送信する
    """
    gas_url = "https://script.google.com/macros/s/AKfycbwbJ_0pFhjxUp5qvu5GLBSZEB1DZfPsrLYqBA4PA_zkxOf8pRN5YYOVCb2iKwTdJcRk2Q/exec"
    
    try:
        # POSTリクエスト用のデータを準備
        post_data = {
            "station_name": station_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "source": "streamlit_app"
        }
        
        # 分析結果がある場合は追加
        if analysis_results:
            post_data.update({
                "analysis_results": {
                    # スプレッドシート列構成に対応したキー名を使用
                    "塾の数": analysis_results.get("塾の数", 0),                    # D列
                    "eスポーツスクールの数": analysis_results.get("eスポーツスクールの数", 0), # E列
                    "小学校の数": analysis_results.get("小学校の数", 0),              # F列
                    "中学校の数": analysis_results.get("中学校の数", 0),              # G列
                    "高校の数": analysis_results.get("高校の数", 0),                # H列
                    "大学の数": analysis_results.get("大学の数", 0),                # I列
                    # 統計・メタデータ
                    "専門学校の数": analysis_results.get("専門学校の数", 0),
                    "幼稚園・保育園の数": analysis_results.get("幼稚園・保育園の数", 0),
                    "その他の数": analysis_results.get("その他の数", 0),
                    "総検索件数": analysis_results.get("総検索件数", 0),
                    "検索半径_km": analysis_results.get("検索半径_km", 0),
                    "分析日時": analysis_results.get("分析日時", pd.Timestamp.now().isoformat())
                }
            })
        
        # ★新規追加: 学校詳細リストがある場合は追加
        if school_details:
            post_data.update({
                "school_details": school_details
            })
        
        # POSTリクエスト送信処理（既存と同じ）
        method_used = "不明"
        try:
            response = requests.post(
                gas_url,
                json=post_data,
                timeout=30
            )
            method_used = "方法1 (json=data)"
        except Exception as method1_error:
            response = requests.post(
                gas_url,
                data=json.dumps(post_data),
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            method_used = "方法2 (data=json)"
        
        # レスポンス処理（既存と同じ）
        if response.status_code == 200:
            try:
                response_json = response.json()
                
                if response_json.get("success"):
                    st.success(f"✅ 分析データを正常に送信し、スプレッドシートに保存されました: {station_name}")
                    return True
                else:
                    error_type = response_json.get("error_type", "unknown")
                    
                    if error_type == "station_not_found":
                        st.warning(f"⚠️ 駅データが見つかりません: {station_name}")
                        with st.expander("💡 解決方法", expanded=True):
                            st.markdown("**以下をお試しください:**")
                            for suggestion in response_json.get("suggestions", []):
                                st.markdown(f"• {suggestion}")
                    else:
                        st.error(f"❌ GAS側でエラーが発生しました: {response_json.get('message', '不明なエラー')}")
                    
                    with st.expander("❌ エラー詳細", expanded=False):
                        st.json(response_json)
                    
                    return False
                    
            except json.JSONDecodeError as json_error:
                st.error(f"❌ レスポンスのJSON解析に失敗しました")
                with st.expander("❌ JSON解析エラー詳細", expanded=True):
                    st.code(f"JSONエラー: {json_error}")
                    st.code(f"生レスポンス: {response.text}")
                return False
        else:
            st.error(f"❌ HTTP エラーが発生しました (ステータス: {response.status_code})")
            return False
            
    except Exception as e:
        st.error(f"❌ 予期しないエラーが発生しました: {str(e)}")
        return False


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
        学校名・塾名を正規化するためのパターンリストを定義
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
            r'株式会社',
            r'有限会社',
            r'教室',
            r'スクール',
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.removal_patterns]

    def normalize(self, name: str) -> str:
        """
        学校名・塾名を正規化して重複の識別を容易にする
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
        学校・塾種別を判定するためのキーワードを定義
        """
        self.school_types = {
            "小学校": ["小学校", "小學校"],
            "中学校": ["中学校", "中學校"],
            "高校": ["高校", "高等学校", "高校"],
            "大学": ["大学", "大学院", "短期大学", "短大"],
            "塾": ["塾", "予備校", "学習塾", "進学塾", "個別指導", "家庭教師", "補習塾", "受験塾", "学習教室"],
            "eスポーツ塾": ["eスポーツ", "ゲーミング", "プロゲーマー", "eSports", "esports", "イースポーツ", "ゲーム教室"],
            "専門学校": ["専門学校", "専門学院", "職業訓練校", "技術学校", "専修学校"],
            "幼稚園・保育園": ["幼稚園", "保育園", "保育所", "こども園", "認定こども園"]
        }
        
        # 教育関連のGoogle Places APIタイプ
        self.education_place_types = {
            'school', 'university', 'primary_school', 'secondary_school', 
            'establishment', 'point_of_interest'
        }
        
        # 明確に除外すべきキーワード（教育関連でないもの）
        self.exclude_keywords = [
            # 商業施設
            'スーパー', 'コンビニ', 'デパート', '百貨店', 'ショッピング', '商店',
            'マーケット', 'ストア', '販売', '小売', 'イオン', 'セブン', 'ローソン',
            
            # 飲食店
            'レストラン', '食堂', 'カフェ', '喫茶', '居酒屋', 'バー', 'ファミレス',
            
            # 医療・福祉
            '病院', 'クリニック', '医院', '歯科', '薬局', 'ドラッグ', '介護',
            
            # その他サービス
            '銀行', '郵便', 'ATM', 'ガソリン', '美容', '理容', 'ホテル', '旅館',
            
            # 公共施設（教育関連でないもの）
            '市役所', '区役所', '警察', '消防', '図書館', '公民館', '体育館',
            
            # 企業・オフィス
            '株式会社', '有限会社', '合同会社', '事務所', 'オフィス', '本社', '支社'
        ]

    def is_education_related(self, name: str, place_types: List[str]) -> bool:
        """
        教育関連施設かどうかを判定
        """
        # 除外キーワードが含まれている場合は教育関連でないと判定
        name_lower = name.lower()
        for exclude_kw in self.exclude_keywords:
            if exclude_kw in name:
                return False
        
        # 教育関連のplace_typesをチェック
        if any(edu_type in place_types for edu_type in self.education_place_types):
            # さらに名前に教育関連キーワードが含まれているかチェック
            education_keywords = [
                '学校', '塾', '教室', '学院', '学園', '大学', '高校', '中学', '小学',
                '幼稚園', '保育園', '専門', '予備校', '学習', '教育', '進学',
                'スクール', 'アカデミー', 'ゼミ', '講座'
            ]
            
            return any(kw in name for kw in education_keywords)
        
        return False

    def classify_school(self, name: str, place_types: List[str]) -> str:
        """
        学校名・塾名と場所タイプ情報から、種別を判定する
        """
        # まず教育関連施設かどうかをチェック
        if not self.is_education_related(name, place_types):
            return None  # 教育関連でない場合はNoneを返す
        
        # eスポーツ塾の判定（優先度高）
        if any(kw in name for kw in self.school_types["eスポーツ塾"]):
            return "eスポーツ塾"
        
        # 幼稚園・保育園の判定
        if any(kw in name for kw in self.school_types["幼稚園・保育園"]):
            return "幼稚園・保育園"
        
        # 専門学校の判定
        if any(kw in name for kw in self.school_types["専門学校"]):
            return "専門学校"
        
        # 大学の判定
        if "university" in place_types or any(kw in name for kw in self.school_types["大学"]):
            return "大学"

        # 塾の判定
        if any(kw in name for kw in self.school_types["塾"]):
            return "塾"

        # 小・中・高の判定
        for school_type in ["小学校", "中学校", "高校"]:
            if any(kw in name for kw in self.school_types[school_type]):
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
        # より具体的な検索除外キーワード
        self.exclude_keywords = [
            "サポート校", "職員室", "事務室", "保健室", 
            "スーパー", "コンビニ", "レストラン", "病院", 
            "銀行", "郵便局", "ガソリンスタンド"
        ]

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
        Google Places API で学校・塾を検索し、結果を返す。
        """
        schools = {}
        # 半径(m)に1.5倍をかけた値でAPIに問い合わせる
        radius_meters = int(radius_km * 1000 * 1.5)
        
        # より具体的な検索キーワード
        search_keywords = [
            # 学校関連（より具体的に）
            "小学校", "中学校", "高等学校", "高校", "大学", "専門学校",
            "幼稚園", "保育園", "こども園",
            
            # 塾関連（より具体的に）
            "学習塾", "進学塾", "予備校", "個別指導塾", "補習塾", "受験塾",
            "英語塾", "数学塾", "国語塾",
            
            # eスポーツ関連
            "eスポーツスクール", "ゲーミングスクール", "プロゲーマー養成",
            "esportsアカデミー", "ゲーム教室"
        ]

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
                            # 学校・塾名を正規化
                            original_name = place['name']
                            normalized_name = self.normalizer.normalize(original_name)

                            if normalized_name:
                                # 学校・塾種別を判定
                                school_type = self.classifier.classify_school(
                                    original_name,
                                    place.get('types', [])
                                )
                                
                                # 教育関連でない場合はスキップ
                                if school_type is None:
                                    continue

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
        学校・塾一覧を種別ごとにまとめて返す
        """
        organized = {
            "小学校": [],
            "中学校": [],
            "高校": [],
            "大学": [],
            "専門学校": [],
            "幼稚園・保育園": [],
            "塾": [],
            "eスポーツ塾": [],
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
        page_title="学校・塾検索アプリ",
        page_icon="🏫",
        layout="wide"
    )
    
    st.title("🏫 駅周辺学校・塾検索アプリ")
    st.markdown("学校、塾、eスポーツ塾を一括検索できます")
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
            help="駅から何km圏内の学校・塾を検索するかを設定してください"
        )
    
    with col2:
        st.metric("設定半径", f"{radius_km:.1f} km")
    
    # ステップ3: 検索実行
    st.header("🔍 ステップ3: 検索実行")
    
    if st.button("🏫 学校・塾を検索する", type="primary", use_container_width=True):
        with st.spinner("学校・塾を検索中..."):
            all_schools = finder.search_all_schools((lat, lng), radius_km)
            organized_results = finder.organize_results(all_schools)
            
            st.session_state.search_results = organized_results
            st.session_state.search_station = station_name
            st.session_state.search_radius = radius_km
        
        # 分析結果を準備してGASに送信
        if organized_results:
            total_schools = sum(len(schools) for schools in organized_results.values())
            
            # スプレッドシートの列構成に対応した分析結果を作成
            analysis_summary = {
                "塾の数": len(organized_results["塾"]),
                "eスポーツスクールの数": len(organized_results["eスポーツ塾"]),
                "小学校の数": len(organized_results["小学校"]),
                "中学校の数": len(organized_results["中学校"]),
                "高校の数": len(organized_results["高校"]),
                "大学の数": len(organized_results["大学"]),
                "専門学校の数": len(organized_results["専門学校"]),
                "幼稚園・保育園の数": len(organized_results["幼稚園・保育園"]),
                "その他の数": len(organized_results["その他"]),
                "総検索件数": total_schools,
                "検索半径_km": radius_km,
                "分析日時": pd.Timestamp.now().isoformat()
            }
            
            # ★新規追加: 学校詳細リストを作成
            school_details = {}
            for category, schools in organized_results.items():
                if schools:  # 空でない場合のみ追加
                    school_details[category] = []
                    for school in schools:
                        school_details[category].append({
                            "name": school['original_name'],
                            "distance": round(school['distance'], 1)
                        })
            
            # 分析データと★学校詳細リストをGASに送信
            with st.spinner("分析データをスプレッドシートに保存中..."):
                send_analysis_data_to_gas(station_name, analysis_summary, school_details)
    
    
    # 結果表示
    if 'search_results' in st.session_state:
        st.header(f"📊 検索結果: {st.session_state.search_station}駅から{st.session_state.search_radius:.1f}km圏内")
        
        results = st.session_state.search_results
        total_schools = sum(len(schools) for schools in results.values())
        
        if total_schools == 0:
            st.warning("該当する学校・塾がありませんでした。検索半径を広げてみてください。")
        else:
            # 概要表示
            cols = st.columns(9)
            with cols[0]:
                st.metric("小学校", len(results["小学校"]))
            with cols[1]:
                st.metric("中学校", len(results["中学校"]))
            with cols[2]:
                st.metric("高校", len(results["高校"]))
            with cols[3]:
                st.metric("大学", len(results["大学"]))
            with cols[4]:
                st.metric("専門学校", len(results["専門学校"]))
            with cols[5]:
                st.metric("幼稚園・保育園", len(results["幼稚園・保育園"]))
            with cols[6]:
                st.metric("塾", len(results["塾"]))
            with cols[7]:
                st.metric("eスポーツ塾", len(results["eスポーツ塾"]))
            with cols[8]:
                st.metric("その他", len(results["その他"]))
            
            st.markdown("---")
            
            # 詳細結果をタブで表示
            tabs = st.tabs([
                "🏫 全体", "🎒 小学校", "📚 中学校", "🎓 高校", "🏛️ 大学", 
                "🔬 専門学校", "👶 幼稚園・保育園", "📖 塾", "🎮 eスポーツ塾", "📋 その他"
            ])
            
            # 全体タブ
            with tabs[0]:
                all_schools_list = []
                for category, schools in results.items():
                    for school in schools:
                        all_schools_list.append({
                            "名称": school['original_name'],
                            "種別": school['type'],
                            "距離 (km)": f"{school['distance']:.1f}"
                        })
                
                if all_schools_list:
                    df = pd.DataFrame(all_schools_list)
                    st.dataframe(df, use_container_width=True)
            
            # 各カテゴリタブ
            categories = ["小学校", "中学校", "高校", "大学", "専門学校", "幼稚園・保育園", "塾", "eスポーツ塾", "その他"]
            for i, category in enumerate(categories):
                with tabs[i + 1]:
                    schools = results[category]
                    if schools:
                        school_data = []
                        for school in schools:
                            school_data.append({
                                "名称": school['original_name'],
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
