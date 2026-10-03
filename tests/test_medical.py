# -*- coding: utf-8 -*-
"""末日地堡生存 —— 医疗救治中心：病例状态机、岗位/设施联动、资源消耗、
危机后健康结算与疫病传播。"""
import pytest

from app.core.database import Base, engine, SessionLocal
from app.core.config import INITIAL_RESOURCES, SURVIVAL_TARGET_DAY
from app.models import GameSession, Resident, Facility, EventLog
from app.services.engine import (
    BunkerEngine,
    BunkerEngineError,
    FACILITY_ZH,
    FOOD,
    WATER,
    POWER,
    OXY,
    CASE_REGISTERED,
    CASE_TREATING,
    CASE_ISOLATED,
    CASE_RECOVERING,
    CRISIS_POOL,
)


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    s = SessionLocal()
    yield s
    s.close()
    Base.metadata.drop_all(bind=engine)


def make_session(db, residents=4, resources=None):
    gs = GameSession(
        name="医疗测试",
        day=1,
        target_day=SURVIVAL_TARGET_DAY,
        status="running",
        resources=resources or dict(INITIAL_RESOURCES),
        survivors=residents,
        score=0,
    )
    db.add(gs)
    db.flush()
    jobs = ["engineer", "farmer", "general", "medic"]
    for i in range(residents):
        db.add(Resident(
            session_id=gs.id, name=f"人{i}", job=jobs[i] if i < len(jobs) else "general",
            health=90, morale=80, alive=1, joined_day=1,
        ))
    for cat in ("power", "farm", "water", "oxygen"):
        db.add(Facility(session_id=gs.id, name=FACILITY_ZH[cat], category=cat,
                        level=1, status="active", built_day=1))
    db.commit()
    db.refresh(gs)
    return gs


class FixedRand:
    def random(self):
        return 0.9  # 不触发危机；疫病传播判定落空

    def choice(self, seq):
        return seq[0]


class SpreadRand:
    """疫病传播必中（0.0 < 传播概率），其余随机判定不触发。"""
    def random(self):
        return 0.0

    def choice(self, seq):
        return seq[0]


def arm_crisis(eng, event_key, target=None):
    event = next(e for e in CRISIS_POOL if e["key"] == event_key)
    crisis = eng._build_crisis(event)
    if target is not None:
        crisis["needs_target"] = True
        crisis["target_id"] = target.id
        crisis["target_name"] = target.name
    eng.session.pending_crisis = crisis
    return crisis


# ---- 容量：医护岗位与医疗救治中心 ----

def test_field_capacity_from_medics_without_clinic(db):
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    # 无救治中心：1 名医护 = 1 个野战救治容量
    assert eng.treatment_capacity() == 1
    gs.residents.append(Resident(session_id=gs.id, name="医2", job="medic",
                                 health=90, morale=80, alive=1, joined_day=1))
    db.commit()
    assert eng.treatment_capacity() == 2


def test_clinic_beds_plus_medics(db):
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.build_facility("clinic")  # Lv.1：2 + 1 = 3 床位
    assert eng.treatment_capacity() == 4  # 3 床位 + 1 医护


def test_hospitalized_medic_not_counted_in_capacity(db):
    """医护本人住院后不再提供人力位（病例不参与生产口径）。"""
    gs = make_session(db, residents=4)
    medic = next(r for r in gs.residents if r.job == "medic")
    eng = BunkerEngine(db, gs, rand=FixedRand())
    # 无中心、1 名医护在岗：容量 1
    assert eng.treatment_capacity() == 1
    medic.health = 45
    eng.medical_register(medic.id, "injury")
    # 唯一的医护已登记住院：野战容量归零
    assert eng.treatment_capacity() == 0


# ---- 手动登记 / 收治 / 隔离 / 出院状态机 ----

def test_manual_register_requires_low_health(db):
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    with pytest.raises(BunkerEngineError):
        eng.medical_register(gs.residents[0].id, "injury")
    gs.residents[0].health = 55
    eng.medical_register(gs.residents[0].id, "injury")
    case = eng._case_of(gs.residents[0].id)
    assert case["stage"] == CASE_REGISTERED
    assert case["type"] == "injury"
    assert case["severity"] == "mild"
    # 重复登记拒绝
    with pytest.raises(BunkerEngineError):
        eng.medical_register(gs.residents[0].id, "disease")


def test_severe_case_classified_below_threshold(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 30
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.medical_register(target.id, "injury")
    assert eng._case_of(target.id)["severity"] == "severe"


def test_unknown_case_type_rejected(db):
    gs = make_session(db)
    gs.residents[0].health = 40
    eng = BunkerEngine(db, gs, rand=FixedRand())
    with pytest.raises(BunkerEngineError):
        eng.medical_register(gs.residents[0].id, "zombie")


def test_treat_consumes_capacity_and_transitions(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.medical_register(target.id, "injury")
    eng.medical_treat(target.id)
    assert eng._case_of(target.id)["stage"] == CASE_TREATING
    # 唯一容量被占用：第二例无法收治
    target2 = gs.residents[1]
    target2.health = 45
    eng.medical_register(target2.id, "injury")
    with pytest.raises(BunkerEngineError):
        eng.medical_treat(target2.id)


def test_isolate_only_for_disease(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.medical_register(target.id, "injury")
    with pytest.raises(BunkerEngineError):
        eng.medical_isolate(target.id)
    # 直接改为疫病病例验证隔离可用
    eng._remove_case(target.id)
    eng._register_case(target.id, "disease", source="测试")
    eng.medical_isolate(target.id)
    assert eng._case_of(target.id)["stage"] == CASE_ISOLATED
    # 隔离不占用治疗容量
    target2 = gs.residents[1]
    target2.health = 45
    eng.medical_register(target2.id, "injury")
    eng.medical_treat(target2.id)
    assert eng._case_of(target2.id)["stage"] == CASE_TREATING


def test_discharge_only_for_recovering(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.medical_register(target.id, "injury")
    with pytest.raises(BunkerEngineError):
        eng.medical_discharge(target.id)


def test_hospitalized_resident_cannot_change_job(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.medical_register(target.id, "injury")
    with pytest.raises(BunkerEngineError):
        eng.set_job(target.id, "farmer")


def test_hospitalized_resident_cannot_join_expedition(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.medical_register(target.id, "injury")
    with pytest.raises(BunkerEngineError):
        eng.send_expedition([target.id], {FOOD: 10, WATER: 10})


# ---- 每日医护结算：自动收治、资源消耗、疗程流转 ----

def test_daily_auto_admit_within_capacity(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.build_facility("clinic")
    eng.medical_register(target.id, "injury")
    eng._apply_medical_care()
    assert eng._case_of(target.id)["stage"] == CASE_TREATING


def test_treating_consumes_resources_and_heals(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.medical_register(target.id, "injury")
    eng.medical_treat(target.id)  # 占用唯一医护容量
    before = dict(gs.resources)
    eng._apply_medical_care()
    # 全套医护消耗：食物 1.0 / 水 1.0 / 电力 1.5
    assert gs.resources[FOOD] == round(before[FOOD] - 1.0, 1)
    assert gs.resources[WATER] == round(before[WATER] - 1.0, 1)
    assert gs.resources[POWER] == round(before[POWER] - 1.5, 1)
    # 治疗回血：基础 8 * (1 + 0.15 医护加成)，另有每日健康结算前的医疗舱逻辑不叠加
    assert target.health > 50
    assert eng._case_of(target.id)["progress"] > 0


def test_rationed_care_when_resources_short(db):
    gs = make_session(db, resources={FOOD: 0, WATER: 0, POWER: 200, OXY: 200})
    target = gs.residents[0]
    target.health = 80
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng._register_case(target.id, "injury", source="测试")
    eng.medical_treat(target.id)
    eng._apply_medical_care()
    # 限额救治：疗效三成 + 虚弱 -2，净回血 8*1.15*0.3 - 2 ≈ 0.76
    assert target.health < 85
    assert target.morale < 80  # 限额士气 -2


def test_unadmitted_registered_case_deteriorates(db):
    gs = make_session(db, residents=6)  # 无中心、1 名医护 → 容量 1
    eng = BunkerEngine(db, gs, rand=FixedRand())
    victims = []
    for r in [x for x in gs.residents if x.job != "medic"][:3]:
        r.health = 45
        eng._register_case(r.id, "injury", source="测试")
        victims.append(r)
    eng._apply_medical_care()
    # 第一名按登记先后自动收治；其余两人未收治，健康恶化 -3
    stages = {r.id: eng._case_of(r.id)["stage"] for r in victims}
    assert sorted(stages.values()).count(CASE_REGISTERED) == 2
    unadmitted = [r for r in victims if stages[r.id] == CASE_REGISTERED]
    assert all(r.health == 42 for r in unadmitted)


def test_treating_progresses_to_recovering_and_discharges(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 60
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.build_facility("clinic")
    eng._register_case(target.id, "injury", source="测试")
    # 直接置为治疗中、疗程 99：下一次结算转入恢复
    case = eng._case_of(target.id)
    case["stage"] = CASE_TREATING
    case["progress"] = 99.0
    eng._apply_medical_care()
    case = eng._case_of(target.id)
    assert case["stage"] == CASE_RECOVERING
    # 恢复观察满 2 天自动出院
    eng._apply_medical_care()
    assert eng._case_of(target.id)["stage"] == CASE_RECOVERING
    eng._apply_medical_care()
    assert eng._case_of(target.id) is None
    assert target.health > 60


def test_isolated_disease_does_not_spread(db):
    gs = make_session(db)
    sick = gs.residents[0]
    eng = BunkerEngine(db, gs, rand=SpreadRand())  # 传播掷点必中
    eng._register_case(sick.id, "disease", source="测试")
    eng.medical_isolate(sick.id)
    eng._apply_medical_care()
    # 隔离中无传播：其他人没有病例
    others = [r for r in gs.residents if r.id != sick.id]
    assert all(eng._case_of(r.id) is None for r in others)


def test_unisolated_disease_spreads_at_most_one_per_day(db):
    gs = make_session(db)
    sick = gs.residents[0]
    eng = BunkerEngine(db, gs, rand=SpreadRand())  # 传播必中
    eng._register_case(sick.id, "disease", source="测试")
    # 保持在登记待收治阶段（容量为 0：唯一医护之外，且本例无中心）
    # 注：sick 不是 medic，容量为 1，会被自动收治；先占满容量
    blocker = gs.residents[2]
    blocker.health = 40
    eng._register_case(blocker.id, "injury", source="测试")
    eng.medical_treat(blocker.id)  # 占满唯一野战容量
    eng._apply_medical_care()
    disease_cases = [
        r for r in gs.residents
        if eng._case_of(r.id) and eng._case_of(r.id)["type"] == "disease"
    ]
    # 原病例 + 至多 1 例新增传染
    assert len(disease_cases) == 2


def test_medic_reduces_spread_chance(db):
    gs = make_session(db, residents=5)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    # 常量口径：每名医护 -2%
    from app.services.engine import CARE_SPREAD_BASE, CARE_SPREAD_MEDIC_REDUCTION
    chance = CARE_SPREAD_BASE - CARE_SPREAD_MEDIC_REDUCTION * eng.job_count("medic")
    assert chance == CARE_SPREAD_BASE - CARE_SPREAD_MEDIC_REDUCTION


def test_case_death_removes_case_and_decrements_survivors(db):
    gs = make_session(db, residents=6)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    victim = gs.residents[0]
    victim.health = 2
    eng._register_case(victim.id, "injury", source="测试")
    # 占满容量使其无法被收治
    blocker = gs.residents[2]
    blocker.health = 40
    eng._register_case(blocker.id, "injury", source="测试")
    eng.medical_treat(blocker.id)
    survivors_before = gs.survivors
    eng._apply_medical_care()  # 未收治 -3：2 → -1 → 死亡
    assert victim.alive == 0
    assert eng._case_of(victim.id) is None
    assert gs.survivors == survivors_before - 1


def test_daily_medical_log_written(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng._register_case(target.id, "injury", source="测试")
    eng._apply_medical_care()
    db.commit()
    logs = db.query(EventLog).filter_by(session_id=gs.id).all()
    assert any("医疗救治中心" in l.title for l in logs)


# ---- 危机后的健康结算 ----

def test_sick_quarantine_registers_isolated_disease_case(db):
    gs = make_session(db)
    target = gs.residents[0]
    eng = BunkerEngine(db, gs, rand=FixedRand())
    arm_crisis(eng, "sick", target=target)
    detail, replayed = eng.resolve_crisis("sick", "quarantine", target_id=target.id)
    assert replayed is False
    case = eng._case_of(target.id)
    assert case is not None
    assert case["type"] == "disease"
    assert case["stage"] == CASE_ISOLATED


def test_sick_public_health_usually_contains_outbreak(db):
    gs = make_session(db)
    target = gs.residents[0]
    eng = BunkerEngine(db, gs, rand=FixedRand())  # random()=0.9 ≥ 0.4：消毒成功
    arm_crisis(eng, "sick", target=target)
    eng.resolve_crisis("sick", "public_health", target_id=target.id)
    assert eng._case_of(target.id) is None


def test_sick_public_health_may_slip_through(db):
    gs = make_session(db)
    target = gs.residents[0]
    eng = BunkerEngine(db, gs, rand=SpreadRand())  # random()=0.0 < 0.4：疏漏
    arm_crisis(eng, "sick", target=target)
    eng.resolve_crisis("sick", "public_health", target_id=target.id)
    case = eng._case_of(target.id)
    assert case is not None and case["type"] == "disease"
    assert case["stage"] == CASE_REGISTERED


def test_crisis_injury_auto_registers_below_threshold(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 55  # raid defend -8 → 47 < 50 自动登记外伤
    eng = BunkerEngine(db, gs, rand=FixedRand())
    arm_crisis(eng, "raid", target=target)
    eng.resolve_crisis("raid", "defend", target_id=target.id)
    case = eng._case_of(target.id)
    assert case is not None and case["type"] == "injury"
    assert case["stage"] == CASE_REGISTERED


def test_crisis_injury_not_registered_above_threshold(db):
    gs = make_session(db)
    target = gs.residents[0]  # health 90 - 8 = 82，无需登记
    eng = BunkerEngine(db, gs, rand=FixedRand())
    arm_crisis(eng, "raid", target=target)
    eng.resolve_crisis("raid", "defend", target_id=target.id)
    assert eng._case_of(target.id) is None


def test_idempotent_replay_does_not_double_register(db):
    gs = make_session(db)
    target = gs.residents[0]
    eng = BunkerEngine(db, gs, rand=FixedRand())
    arm_crisis(eng, "sick", target=target)
    eng.resolve_crisis("sick", "quarantine", target_id=target.id)
    # 同一结算重放：不重复登记、不产生第二个病例
    detail, replayed = eng.resolve_crisis("sick", "quarantine", target_id=target.id)
    assert replayed is True
    assert len(eng._cases()) == 1


# ---- 探索队 / 押运队返程后的健康结算 ----

def test_expedition_return_registers_wounded_member(db):
    gs = make_session(db)
    victim = gs.residents[0]
    victim.health = 60
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.send_expedition([victim.id], {FOOD: 20, WATER: 20})
    # 返程途中受伤：模拟遭遇效果把健康打到阈值以下
    victim.health = 40
    eng._settle_expedition(gs.expedition, reason="测试返程")
    case = eng._case_of(victim.id)
    assert case is not None and case["type"] == "injury"


# ---- 终局收敛：在册病例清空 ----

def test_finish_clears_medical_cases(db):
    gs = make_session(db)
    target = gs.residents[0]
    target.health = 50
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng._register_case(target.id, "injury", source="测试")
    gs.day = gs.target_day
    eng._check_end()
    assert gs.status == "win"
    assert gs.medical is None
