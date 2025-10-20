from flask import Flask, render_template, request, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func, inspect, text
import uuid, re, secrets, string

# ================== Flask 基本設定 ==================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace_me_with_random_secret'      # 実運用は環境変数で
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# ================== モデル ==================
class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    email    = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    # 将来用: 役割/状態が欲しければ下の2行のコメントを外す
    # role      = db.Column(db.String(20), default='member')
    # is_active = db.Column(db.Boolean, default=True)


class Invite(db.Model):
    """管理者が発行する受験用URL（トークン）"""
    id     = db.Column(db.Integer, primary_key=True)
    token  = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    used   = db.Column(db.Boolean, default=False)
    # ★ 所有者（発行者）
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)


class Applicant(db.Model):
    """受験提出時に自動作成される応募者"""
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(150), nullable=False)
    email      = db.Column(db.String(150), nullable=False)
    score      = db.Column(db.Float, nullable=True)   # 総合スコア（回答平均）
    risk       = db.Column(db.Float, nullable=True)   # 0-100（高いほどリスク高）
    exam_token = db.Column(db.String(36), unique=True, nullable=False)  # Invite.token を流用
    # ★ 所有者（だれの案件か）
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)


class Skill(db.Model):
    id     = db.Column(db.Integer, primary_key=True)
    name   = db.Column(db.String(100), nullable=False)
    # ★ 所有者
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)


class Question(db.Model):
    """問題（自由記述:free / 選択式:mc）"""
    id      = db.Column(db.Integer, primary_key=True)
    text    = db.Column(db.Text, nullable=False)
    skill_id= db.Column(db.Integer, db.ForeignKey('skill.id'))
    qtype   = db.Column(db.String(20), default='free')  # 'free' or 'mc'

    # 自由記述用の採点設定
    model_answer     = db.Column(db.Text, nullable=True)      # 模範解答（任意）
    keywords         = db.Column(db.Text, nullable=True)      # "kw1, kw2, ..."
    weight_structure = db.Column(db.Float, default=50.0)      # 構造の重み(%)
    weight_keywords  = db.Column(db.Float, default=30.0)      # キーワード一致の重み(%)
    weight_similarity= db.Column(db.Float, default=20.0)      # 模範解答との類似度の重み(%)

    # ★ 所有者
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)


class Answer(db.Model):
    """選択式の選択肢"""
    id          = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'))
    text        = db.Column(db.Text, nullable=False)
    is_correct  = db.Column(db.Boolean, default=False)


class ApplicantResponse(db.Model):
    """応募者の回答（free と mc で共通）"""
    id                 = db.Column(db.Integer, primary_key=True)
    applicant_id       = db.Column(db.Integer, db.ForeignKey('applicant.id'), nullable=False)
    question_id        = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
    answer_text        = db.Column(db.Text, nullable=True)    # free用
    selected_answer_id = db.Column(db.Integer, db.ForeignKey('answer.id'), nullable=True)  # mc用
    score              = db.Column(db.Float, nullable=True)   # 0-100


# ================== Login 管理 ==================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ================== 表示用フィルタ（退職リスク） ==================
@app.template_filter('risk_grade')
def risk_grade(risk):
    """
    数値(0-100) → S/A/B/C/D に変換。
    0が低リスク、100が高リスク想定。
    """
    if risk is None:
        return '-'
    r = float(risk)
    if r < 10:
        return 'S'
    if r < 25:
        return 'A'
    if r < 50:
        return 'B'
    if r < 75:
        return 'C'
    return 'D'

@app.template_filter('fmt_pct')
def fmt_pct(value):
    return '-' if value is None else f"{round(float(value), 1)}%"


# ================== 採点ロジック（外部API不要） ==================
def _normalize(text: str) -> list[str]:
    t = (text or "").lower()
    t = re.sub(r'[^a-z0-9ぁ-んァ-ン一-龥\s]', ' ', t)  # 英数と日本語以外の記号を空白に
    toks = [w for w in t.split() if w]
    return toks  # ←重要

def _jaccard(a: str, b: str) -> float:
    A, B = set(_normalize(a)), set(_normalize(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _keyword_score(ans: str, keywords_csv: str) -> float:
    if not keywords_csv:
        return 0.0
    kws = [k.strip().lower() for k in keywords_csv.split(',') if k.strip()]
    if not kws:
        return 0.0
    toks = set(_normalize(ans))
    hit = sum(1 for k in kws if k in toks)
    return 100.0 * hit / len(kws)

def _grade_sentence_structure_rule_based(text: str) -> float:
    """Sentence Structure の簡易ルール採点（0-100）"""
    t = (text or "").strip()
    if not t:
        return 0.0
    sentences = [s.strip() for s in re.split(r'[\.!?。\n]+', t) if s.strip()]
    n = len(sentences)
    length_list = [len(s.split()) for s in sentences] if n else [0]
    avg_len = sum(length_list) / max(n, 1)
    var_len = (sum((x - avg_len) ** 2 for x in length_list) / max(n, 1)) ** 0.5
    commas = t.count(',') + t.count('，') + t.count('、')
    semis = t.count(';')
    caps_ratio = sum(1 for s in sentences if s[:1].isupper()) / max(n, 1)

    score = 0.0
    score += 20 if 2 <= n <= 7 else max(0, 20 - abs(n - 4) * 5)
    score += 20 if 8 <= avg_len <= 20 else max(0, 20 - abs(avg_len - 14) * 2)
    score += 15 if 4 <= var_len <= 12 else max(0, 15 - abs(var_len - 8) * 2)
    punct = min(commas + semis, 6)
    score += (punct / 6) * 15
    score += caps_ratio * 15
    long_penalty = sum(1 for L in length_list if L > 30)
    score += max(0, 15 - long_penalty * 7.5)
    return max(0.0, min(100.0, round(score, 1)))

def grade_free_answer(ans_text: str, q: 'Question') -> float:
    """構造 + キーワード + 模範解答類似度 を重み付き合成（合計は自動正規化）"""
    ws = max(0.0, q.weight_structure or 0.0)
    wk = max(0.0, q.weight_keywords  or 0.0)
    wi = max(0.0, q.weight_similarity or 0.0)
    total = ws + wk + wi
    if total <= 0:
        ws, wk, wi, total = 50.0, 30.0, 20.0, 100.0
    s_struct = _grade_sentence_structure_rule_based(ans_text)
    s_kw    = _keyword_score(ans_text, q.keywords or "")
    s_sim   = _jaccard(ans_text, q.model_answer or "") * 100.0
    score = (s_struct * ws + s_kw * wk + s_sim * wi) / total
    return round(score, 1)

def grade_mc(selected_ans: 'Answer') -> float:
    return 100.0 if (selected_ans and selected_ans.is_correct) else 0.0


# ================== 共通ユーティリティ（所有者スコープ） ==================
def _owner_filter(query, model):
    """owner_user_id を持つテーブルなら現在ユーザーのデータのみに絞る。"""
    if hasattr(model, "owner_user_id"):
        return query.filter(getattr(model, "owner_user_id") == current_user.id)
    return query

def _ensure_owner(obj):
    """レコードが自分の所有物であることを確認。違えば 404。"""
    if hasattr(obj, "owner_user_id"):
        if obj.owner_user_id != current_user.id:
            abort(404)


# ================== 認証 ==================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('メールアドレスまたはパスワードが間違っています')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ================== 画面・操作（全て所有者スコープ） ==================
@app.route('/dashboard')
@login_required
def dashboard():
    invites   = _owner_filter(db.session.query(Invite), Invite).order_by(Invite.id.desc()).all()
    applicants= _owner_filter(db.session.query(Applicant), Applicant).order_by(Applicant.id.desc()).all()
    return render_template('dashboard.html', applicants=applicants, invites=invites)

@app.route('/manage')
@login_required
def manage():
    skills    = _owner_filter(db.session.query(Skill), Skill).all()
    questions = _owner_filter(db.session.query(Question), Question).order_by(Question.id.desc()).all()
    # 選択肢マップ
    answers_map = {}
    for q in questions:
        if q.qtype == 'mc':
            answers_map[q.id] = Answer.query.filter_by(question_id=q.id).all()
    return render_template('manage.html', skills=skills, questions=questions, answers_map=answers_map)

# テストURL発行（招待を1件作成）
@app.route('/issue_test', methods=['POST'])
@login_required
def issue_test():
    inv = Invite(owner_user_id=current_user.id)
    db.session.add(inv)
    db.session.commit()
    flash(f'テストURLを発行しました: {url_for("take_exam", token=inv.token, _external=True)}')
    return redirect(url_for('dashboard'))

# 発行済みURLの削除
@app.route('/delete_invite/<int:id>', methods=['POST'])
@login_required
def delete_invite(id):
    inv = Invite.query.get_or_404(id)
    _ensure_owner(inv)
    db.session.delete(inv)
    db.session.commit()
    flash('発行済みURLを削除しました')
    return redirect(url_for('dashboard'))

# スキル追加/削除
@app.route('/add_skill', methods=['POST'])
@login_required
def add_skill():
    name = (request.form.get('name') or '').strip()
    if name:
        db.session.add(Skill(name=name, owner_user_id=current_user.id))
        db.session.commit()
        flash('スキルを追加しました')
    else:
        flash('スキル名は必須です')
    return redirect(url_for('manage'))

@app.route('/delete_skill/<int:id>')
@login_required
def delete_skill(id):
    s = Skill.query.get_or_404(id)
    _ensure_owner(s)
    db.session.delete(s)
    db.session.commit()
    flash('スキルを削除しました')
    return redirect(url_for('manage'))

# 問題追加（自由記述）
@app.route('/add_question_free', methods=['POST'])
@login_required
def add_question_free():
    text = (request.form.get('text') or '').strip()
    skill_id = request.form.get('skill_id')
    model_answer = (request.form.get('model_answer') or '').strip()
    keywords = (request.form.get('keywords') or '').strip()
    ws = float(request.form.get('weight_structure', 50) or 50)
    wk = float(request.form.get('weight_keywords', 30) or 30)
    wi = float(request.form.get('weight_similarity', 20) or 20)
    if text and skill_id:
        q = Question(
            text=text, skill_id=int(skill_id), qtype='free',
            model_answer=model_answer or None,
            keywords=keywords or None,
            weight_structure=ws, weight_keywords=wk, weight_similarity=wi,
            owner_user_id=current_user.id
        )
        db.session.add(q)
        db.session.commit()
        flash('（自由記述）問題を追加しました')
    else:
        flash('問題文とスキルは必須です')
    return redirect(url_for('manage'))

# 問題追加（選択式）
@app.route('/add_question_mc', methods=['POST'])
@login_required
def add_question_mc():
    text = (request.form.get('text_mc') or '').strip()
    skill_id = request.form.get('skill_id_mc')
    options_raw = (request.form.get('options_mc') or '').strip()
    if text and skill_id and options_raw:
        q = Question(text=text, skill_id=int(skill_id), qtype='mc', owner_user_id=current_user.id)
        db.session.add(q)
        db.session.flush()  # q.id を確保
        # 1行1選択肢、先頭に * を付けると正答
        for line in options_raw.splitlines():
            opt = line.strip()
            if not opt:
                continue
            is_correct = False
            if opt.startswith('*'):
                is_correct = True
                opt = opt[1:].strip()
            db.session.add(Answer(question_id=q.id, text=opt, is_correct=is_correct))
        db.session.commit()
        flash('（選択式）問題を追加しました')
    else:
        flash('問題文・スキル・選択肢は必須です')
    return redirect(url_for('manage'))

# 問題編集（free / mc 共通）
@app.route('/edit_question/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_question(id):
    q = Question.query.get_or_404(id)
    _ensure_owner(q)
    if request.method == 'POST':
        q.text = (request.form.get('text') or q.text).strip()
        q.skill_id = int(request.form.get('skill_id', q.skill_id))
        if q.qtype == 'free':
            q.model_answer = (request.form.get('model_answer') or '').strip() or None
            q.keywords = (request.form.get('keywords') or '').strip() or None
            q.weight_structure = float(request.form.get('weight_structure', q.weight_structure) or q.weight_structure)
            q.weight_keywords  = float(request.form.get('weight_keywords',  q.weight_keywords)  or q.weight_keywords)
            q.weight_similarity= float(request.form.get('weight_similarity',q.weight_similarity)or q.weight_similarity)
        db.session.commit()
        flash('問題を更新しました')
        return redirect(url_for('manage'))
    skills = _owner_filter(db.session.query(Skill), Skill).all()
    options = Answer.query.filter_by(question_id=q.id).all() if q.qtype == 'mc' else []
    return render_template('edit_question.html', q=q, skills=skills, options=options)

# MC選択肢の追加・削除
@app.route('/add_option/<int:qid>', methods=['POST'])
@login_required
def add_option(qid):
    q = Question.query.get_or_404(qid)
    _ensure_owner(q)
    if q.qtype != 'mc':
        flash('この問題は選択式ではありません')
        return redirect(url_for('edit_question', id=qid))
    text_opt = (request.form.get('text') or '').strip()
    is_correct = True if request.form.get('is_correct') == 'on' else False
    if text_opt:
        db.session.add(Answer(question_id=q.id, text=text_opt, is_correct=is_correct))
        db.session.commit()
        flash('選択肢を追加しました')
    else:
        flash('選択肢の本文は必須です')
    return redirect(url_for('edit_question', id=qid))

@app.route('/delete_option/<int:opt_id>')
@login_required
def delete_option(opt_id):
    opt = Answer.query.get_or_404(opt_id)
    q = Question.query.get_or_404(opt.question_id)
    _ensure_owner(q)
    db.session.delete(opt)
    db.session.commit()
    flash('選択肢を削除しました')
    return redirect(url_for('edit_question', id=q.id))

@app.route('/delete_question/<int:id>')
@login_required
def delete_question(id):
    q = Question.query.get_or_404(id)
    _ensure_owner(q)
    if q.qtype == 'mc':
        Answer.query.filter_by(question_id=q.id).delete()
    db.session.delete(q)
    db.session.commit()
    flash('問題を削除しました')
    return redirect(url_for('manage'))

# 応募者詳細（所有者のみ）
@app.route('/applicant/<int:id>')
@login_required
def applicant_detail(id):
    applicant = Applicant.query.get_or_404(id)
    _ensure_owner(applicant)

    rows = (
        db.session.query(func.coalesce(func.min(Skill.name), '-'), func.avg(ApplicantResponse.score))
        .join(Question, Question.id == ApplicantResponse.question_id)
        .outerjoin(Skill, Skill.id == Question.skill_id)
        .filter(ApplicantResponse.applicant_id == applicant.id)
        .group_by(Question.skill_id)
        .all()
    )
    skill_percentages = {name: round(avg or 0, 1) for name, avg in rows}

    responses = db.session.query(
        Question.text, ApplicantResponse.answer_text, ApplicantResponse.score, Skill.name,
        ApplicantResponse.selected_answer_id
    ).join(Question, ApplicantResponse.question_id == Question.id
    ).outerjoin(Skill, Question.skill_id == Skill.id
    ).filter(ApplicantResponse.applicant_id == applicant.id).all()

    sel_text_map = {}
    sel_ids = [r.selected_answer_id for r in responses if r.selected_answer_id]
    if sel_ids:
        for a in Answer.query.filter(Answer.id.in_(sel_ids)).all():
            sel_text_map[a.id] = a.text

    return render_template('applicant_detail.html',
                           applicant=applicant,
                           skill_percentages=skill_percentages,
                           responses=responses,
                           sel_text_map=sel_text_map)

# 受験ページ（トークンのオーナーにひも付く問題のみ）
@app.route('/exam/<token>', methods=['GET', 'POST'])
def take_exam(token):
    invite = Invite.query.filter_by(token=token).first_or_404()
    owner_id = invite.owner_user_id

    qset = db.session.query(Question)
    if owner_id is not None:
        qset = qset.filter(Question.owner_user_id == owner_id)
    questions = qset.order_by(Question.id.asc()).all()

    options_map = {q.id: Answer.query.filter_by(question_id=q.id).all() for q in questions if q.qtype == 'mc'}
    applicant = Applicant.query.filter_by(exam_token=token).first()

    if request.method == 'POST':
        if applicant is None:
            name = (request.form.get('name') or '').strip()
            email = (request.form.get('email') or '').strip()
            if not name or not email:
                flash('氏名とメールは必須です')
                return redirect(url_for('take_exam', token=token))
            applicant = Applicant(name=name, email=email, exam_token=token, owner_user_id=owner_id)
            db.session.add(applicant)
            invite.used = True

        db.session.flush()  # applicant.id を確保
        for q in questions:
            if q.qtype == 'free':
                ans = (request.form.get(f'answer_free_{q.id}') or '').strip()
                s = grade_free_answer(ans, q)
                db.session.add(ApplicantResponse(
                    applicant_id=applicant.id,
                    question_id=q.id,
                    answer_text=ans,
                    selected_answer_id=None,
                    score=s
                ))
            else:
                selected_id = request.form.get(f'answer_mc_{q.id}')
                selected = Answer.query.get(int(selected_id)) if selected_id else None
                s = grade_mc(selected)
                db.session.add(ApplicantResponse(
                    applicant_id=applicant.id,
                    question_id=q.id,
                    answer_text=None,
                    selected_answer_id=selected.id if selected else None,
                    score=s
                ))

        # 総合スコア/リスク
        db.session.flush()
        resps = ApplicantResponse.query.filter_by(applicant_id=applicant.id).all()
        if resps:
            applicant.score = round(sum(r.score for r in resps if r.score is not None) / len(resps), 1)
            applicant.risk  = round(max(0.0, 100.0 - applicant.score), 1)

        db.session.commit()
        flash('回答を送信しました。ありがとうございました。')
        return redirect(url_for('take_exam', token=token))

    return render_template('exam.html', invite=invite, applicant=applicant,
                           questions=questions, options_map=options_map)


# ========== 管理者向けユーザー管理（簡易: admin@example.com を管理者扱い） ==========
def admin_required(view):
    from functools import wraps
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        # role カラムが無い前提の簡易判定：メールが admin@example.com なら管理者扱い
        is_admin = (getattr(current_user, 'email', '') == 'admin@example.com')
        # 将来 role を付けるなら:
        # role = getattr(current_user, 'role', None)
        # is_admin = is_admin or role in ('admin', 'owner')
        if not is_admin:
            flash('このページにアクセスする権限がありません', 'error')
            return redirect(url_for('dashboard'))
        return view(*args, **kwargs)
    return wrapper

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    kw = (request.args.get('q') or '').strip().lower()
    qset = db.session.query(User)
    if kw:
        qset = qset.filter(User.email.ilike(f'%{kw}%'))
    users = qset.order_by(User.id.asc()).all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/new', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_users_new():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password_plain = (request.form.get('initial_password') or '').strip()
        if not email:
            flash('メールは必須です', 'error')
            return redirect(url_for('admin_users_new'))
        if db.session.query(User).filter_by(email=email).first():
            flash('このメールは既に登録されています', 'error')
            return redirect(url_for('admin_users_new'))
        if not password_plain:
            alphabet = string.ascii_letters + string.digits
            password_plain = ''.join(secrets.choice(alphabet) for _ in range(10))
        user = User(email=email, password=generate_password_hash(password_plain))
        db.session.add(user)
        db.session.commit()
        flash(f'ユーザーを作成しました（初期PW: {password_plain}）', 'success')
        return redirect(url_for('admin_users'))
    return render_template('admin_users_new.html')


# ================== 簡易オートマイグレーション（不足カラムの追加） ==================
def _ensure_new_columns():
    """
    既存の database.db に新カラムが無い場合は ALTER TABLE で追加。
    """
    try:
        insp = inspect(db.engine)

        # 共通: owner_user_id を足す
        def _add_owner_if_missing(table):
            if insp.has_table(table):
                cols = {c['name'] for c in insp.get_columns(table)}
                if 'owner_user_id' not in cols:
                    db.session.execute(text(f"ALTER TABLE {table} ADD COLUMN owner_user_id INTEGER"))
                    return True
            return False

        touched = []
        for t in ('invite', 'applicant', 'skill', 'question'):
            if _add_owner_if_missing(t):
                touched.append(t)
        if touched:
            db.session.commit()
            print("[auto-migrate] owner_user_id added to:", touched)

        # Question の採点用カラム
        if insp.has_table('question'):
            qcols = {c['name'] for c in insp.get_columns('question')}
            stmts = []
            if 'model_answer' not in qcols:
                stmts.append("ALTER TABLE question ADD COLUMN model_answer TEXT")
            if 'keywords' not in qcols:
                stmts.append("ALTER TABLE question ADD COLUMN keywords TEXT")
            if 'weight_structure' not in qcols:
                stmts.append("ALTER TABLE question ADD COLUMN weight_structure REAL DEFAULT 50")
            if 'weight_keywords' not in qcols:
                stmts.append("ALTER TABLE question ADD COLUMN weight_keywords REAL DEFAULT 30")
            if 'weight_similarity' not in qcols:
                stmts.append("ALTER TABLE question ADD COLUMN weight_similarity REAL DEFAULT 20")
            if 'qtype' not in qcols:
                stmts.append("ALTER TABLE question ADD COLUMN qtype VARCHAR(20) DEFAULT 'free'")
            for s in stmts:
                db.session.execute(text(s))
            if stmts:
                db.session.commit()
                print("[auto-migrate] question columns added:", stmts)

        # ApplicantResponse の selected_answer_id
        if insp.has_table('applicant_response'):
            rcols = {c['name'] for c in insp.get_columns('applicant_response')}
            if 'selected_answer_id' not in rcols:
                db.session.execute(text("ALTER TABLE applicant_response ADD COLUMN selected_answer_id INTEGER"))
                db.session.commit()
                print("[auto-migrate] applicant_response selected_answer_id added")

    except Exception as e:
        print("[auto-migrate] skipped or failed:", e)


# ================== 起動ブロック ==================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        _ensure_new_columns()

        # 初期管理ユーザー（無ければ）
        if not User.query.filter_by(email="admin@example.com").first():
            hashed = generate_password_hash("password123")
            db.session.add(User(email="admin@example.com", password=hashed))
            db.session.commit()

    app.run(debug=True)

