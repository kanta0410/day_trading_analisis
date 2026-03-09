import pandas as pd
import glob
import os
import sys
import google.generativeai as genai
from datetime import datetime
from tabulate import tabulate
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_and_clean_csv(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    all_trades = []
    
    for f in files:
        print(f"Checking file: {f}")
        for enc in ['cp932', 'utf-8-sig', 'utf-8']:
            try:
                df = pd.read_csv(f, encoding=enc)
                if df.empty: continue
                
                # Case 1: Realized P/L CSV (実現損益)
                pl_col = None
                for col in df.columns:
                    if '実現損益' in col and '(円)' in col:
                        pl_col = col
                        break
                
                if pl_col:
                    print(f"  Detected Realized P/L format in {os.path.basename(f)}")
                    df_pl = df.dropna(subset=[pl_col, '銘柄名'])
                    for _, row in df_pl.iterrows():
                        try:
                            pl_val = str(row[pl_col]).replace(',', '').replace('+', '')
                            if pl_val == '-' or pl_val.strip() == '': continue
                            all_trades.append({
                                '銘柄名': str(row['銘柄名']).strip(),
                                '損益': float(pl_val),
                                '約定日': pd.to_datetime(row['約定日'])
                            })
                        except:
                            continue
                    break 
                
                # Case 2: Execution Details (約定明細)
                if '約定日' in df.columns and '取引' in df.columns:
                    print(f"  Detected Execution Details format in {os.path.basename(f)}")
                    matched = process_matched_trades(df)
                    all_trades.extend(matched)
                    break
                    
            except Exception as e:
                continue
                
    return pd.DataFrame(all_trades)

def process_matched_trades(df):
    def clean_num(x):
        if pd.isna(x): return 0
        if isinstance(x, str):
            res = x.replace(',', '')
            return float(res) if res != '-' and res.strip() != '' else 0
        return float(x)

    df = df.copy()
    qty_col = '約定数量(株/口)' if '約定数量(株/口)' in df.columns else '約定数量'
    price_col = '約定単価(円)' if '約定単価(円)' in df.columns else '約定単価'
    
    if qty_col not in df.columns or price_col not in df.columns:
        return []

    df['qty_clean'] = df[qty_col].apply(clean_num)
    df['price_clean'] = df[price_col].apply(clean_num)
    df['date_clean'] = pd.to_datetime(df['約定日'])
    df = df.sort_values('date_clean')
    
    results = []
    # Use only '銘柄名' for grouping to avoid tuple issues
    for symbol, group in df.groupby('銘柄名'):
        symbol_name = str(symbol).strip()
        inventory = []
        for _, row in group.iterrows():
            side_type = str(row['取引'])
            action = str(row.get('売買', ''))
            qty = row['qty_clean']
            price = row['price_clean']
            date = row['date_clean']
            
            if '新規' in side_type or (side_type in ['買付', '売付'] and not ('埋' in action or '返済' in side_type)):
                side = 'Long' if ('買' in action or '買付' in side_type) else 'Short'
                inventory.append({'qty': qty, 'price': price, 'side': side, 'date': date})
            elif '返済' in side_type or '埋' in action:
                while qty > 0 and inventory:
                    match = inventory[0]
                    take_qty = min(qty, match['qty'])
                    if match['side'] == 'Long':
                        profit = (price - match['price']) * take_qty
                    else:
                        profit = (match['price'] - price) * take_qty
                    
                    results.append({
                        '銘柄名': symbol_name,
                        '損益': profit,
                        '約定日': date
                    })
                    qty -= take_qty
                    match['qty'] -= take_qty
                    if match['qty'] <= 0:
                        inventory.pop(0)
    return results

def calculate_stats(df_all):
    if df_all.empty:
        return None
        
    stats = {}
    total_profit = df_all[df_all['損益'] > 0]['損益'].sum()
    total_loss = abs(df_all[df_all['損益'] < 0]['損益'].sum())
    
    stats['total_trades'] = len(df_all)
    stats['win_count'] = len(df_all[df_all['損益'] > 0])
    stats['win_rate'] = stats['win_count'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
    stats['pf'] = total_profit / total_loss if total_loss > 0 else float('inf')
    stats['avg_win'] = total_profit / stats['win_count'] if stats['win_count'] > 0 else 0
    stats['avg_loss'] = total_loss / (stats['total_trades'] - stats['win_count']) if (stats['total_trades'] - stats['win_count']) > 0 else 0
    
    symbol_stats = []
    for name, group in df_all.groupby('銘柄名'):
        trade_count = len(group)
        if trade_count == 0: continue
        ws = len(group[group['損益'] > 0])
        wr = ws / trade_count
        tp = group[group['損益'] > 0]['損益'].sum()
        tl = abs(group[group['損益'] < 0]['損益'].sum())
        p_factor = tp / tl if tl > 0 else float('inf')
        total_pl = group['損益'].sum()
        
        score = (wr * 50) + (min(p_factor, 10) * 5) + (1 if total_pl > 0 else -1)
        
        symbol_stats.append({
            '銘柄': name,
            '勝率': f"{wr:.1%}",
            'PF': f"{p_factor:.2f}",
            '損益計': int(total_pl),
            'トレード数': trade_count,
            'score': score
        })
    
    stats['symbols'] = sorted(symbol_stats, key=lambda x: x['score'], reverse=True)
    df_all['date_only'] = df_all['約定日'].dt.date
    daily_pl = df_all.groupby('date_only')['損益'].sum().reset_index()
    daily_pl.columns = ['日付', '損益']
    stats['daily_pl'] = daily_pl.sort_values('日付')
    return stats

def get_ai_review(stats):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "AI review unavailable (No API Key set)."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        summary = f"""
        トレード統計:
        - 総トレード数: {stats['total_trades']}
        - 勝率: {stats['win_rate']:.1%}
        - プロフィットファクター (PF): {stats['pf']:.2f}
        - 平均利益: {stats['avg_win']:.0f}円
        - 平均損失: {stats['avg_loss']:.0f}円
        - 最も相性の良い銘柄: {stats['symbols'][0]['銘柄'] if stats['symbols'] else 'N/A'}
        - 最も相性の悪い銘柄: {stats['symbols'][-1]['銘柄'] if stats['symbols'] else 'N/A'}
        """
        prompt = f"伝説の厳しいプロトレーダーとして、以下の結果に冷徹にダメ出しをしてください。お世辞抜き、辛口で。日本語。\n\n{summary}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Review Error: {e}"

def generate_report(stats, ai_review):
    report = f"# 📈 トレード分析レポート ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
    report += "## 📊 総合パフォーマンス\n"
    report += f"- **勝率:** {stats['win_rate']:.1%}\n"
    report += f"- **プロフィットファクター (PF):** {stats['pf']:.2f}\n"
    report += f"- **総損益:** {stats['daily_pl']['損益'].sum():,.0f}円\n"
    report += f"- **総トレード数:** {stats['total_trades']}\n\n"
    
    report += "## 📅 損益カレンダー\n"
    cal_df = stats['daily_pl'].copy()
    cal_df['損益'] = cal_df['損益'].apply(lambda x: f"{x:,.0f}円")
    report += tabulate(cal_df, headers='keys', tablefmt="github", showindex=False)
    report += "\n\n"
    
    report += "## 🏆 銘柄別相性ランキング\n"
    report += "### 🚀 Top 5 (相性の良い銘柄)\n"
    report += tabulate([{k:v for k,v in x.items() if k != 'score'} for x in stats['symbols'][:5]], headers="keys", tablefmt="github")
    report += "\n\n### 💀 Bottom 5 (相性の悪い銘柄)\n"
    report += tabulate([{k:v for k,v in x.items() if k != 'score'} for x in stats['symbols'][-5:][::-1]], headers="keys", tablefmt="github")
    report += "\n\n"
    
    report += "## 👺 プロトレーダーによる冷徹な反省文\n"
    report += ai_review
    return report

def send_email(report_text):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")
    
    if not all([sender, password, receiver]):
        print("Email settings missing. Skipping email notification.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = f"【トレード分析レポート】{datetime.now().strftime('%Y/%m/%d')}"
    
    msg.attach(MIMEText(report_text, 'plain'))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"Email sent to {receiver}")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    raw_df = load_and_clean_csv("trades")
    stats = calculate_stats(raw_df)
    
    if stats:
        review = get_ai_review(stats)
        report = generate_report(stats, review)
        with open("trade_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("Trade report generated: trade_report.md")
        send_email(report)
    else:
        print("No trades found.")
