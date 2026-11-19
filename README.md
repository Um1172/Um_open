[README.md](https://github.com/user-attachments/files/35063739/README.md)
# クリスタルLM (CrystalLM)

Windows PC 上の **LM Studio** と連携して自律稼働する、多層永続メモリ・チャット履歴保存対応のAIエージェント「クリスタルLM（CrystalLM）」です。

## 主な機能
1. **Gemini / ChatGPT 風のモダンなダークモード3ペインUI (`static/index.html`)**
   - **左サイドバー**: 新規チャット作成、過去スレッドの切り替え・削除（`sessions/` に自動保存）、および永続メモリファイル（`rule.md`, `profile.md`, `memory.md`, `plan.md`, `work_log.md`）の閲覧・手動編集・保存モーダル。
   - **中央チャットエリア**: Markdown レンダリング（`marked.js`）、コードブロックのシンタックスハイライト（`highlight.js`）＋コピーボタン、`Enter` で送信 / `Shift+Enter` で改行。
   - **右インスペクター**: 進行中の `plan.md` チェックリストと `workspace/` 内の生成ファイル一覧（クリックでプレビュー）。
2. **Thinking トークン・作業ログの折りたたみ収納**
   - `<think>...</think>` 内の思考プロセスやツール実行履歴は、回答の直前に「**思考プロセス・作業ログを表示 ▼**」アコーディオンとしてデフォルト閉じた状態で収納されます。
3. **UI からのモデル切り替え**
   - ヘッダーのモデル選択ドロップダウンと「更新」ボタン（`GET /api/models`）から、LM Studio で利用可能なモデルを切り替えて推論に反映できます。
4. **多層 Markdown 永続メモリシステム (`memory/`)**
   - `memory/rule.md`: エージェントの基本行動指針・恒久的指示
   - `memory/profile.md`: ユーザー情報・PC環境・好み
   - `memory/memory.md`: 会話で得られた重要な決定事項・知識の要約
   - `memory/plan.md`: 進行中タスクの計画チェックリスト
   - `memory/work_log.md`: ツール実行履歴・作成ファイル・作業記録
   - 毎回の推論時にシステムプロンプトへ自動注入され、エージェントの `update_memory` ツールおよびターン終了時の記憶整理ステップにより自動更新されます。

## 起動方法
1. **LM Studio** を起動し、モデルをロードして **Local Server (`http://localhost:1234`)** を開始します。
2. [start.bat](file:///c:/anch/start.bat) をダブルクリックします。
3. 自動的にサーバーが起動し、ブラウザで `http://localhost:8000` が開きます。
