% =========================================================
% advisor.pl — советчик "лучшего действия прямо сейчас"
% ---------------------------------------------------------
% Назначение:
%   Жадный советчик для упрощённого Hearthstone-движка из hearthstone.pl.
%   Делает ровно ОДИН лучший шаг "здесь и сейчас" без глубокого поиска.
%
% Основа:
%   Работает поверх уже загруженного hearthstone.pl (ничего в нём не меняет).
%   Все тяжёлые действия движка (play_minion/3, attack/4, end_turn/2)
%   вызываются РОВНО один раз и только после выбора действия.
%
% Приоритеты (по убыванию):
%   1) KILL — добить вражеского миньона, если хватает атаки (A >= HP цели)
%      Ранжирование цели: 2*HP + ATK (+5, если есть taunt).
%      Атакующий должен быть ГОТОВ (CanAttack=true).
%   2) FACE — ударить в лицо, если на столе оппонента нет таунтов.
%      Берём моего ГОТОВОГО атакующего с максимальной атакой (>0) и без RushOnly.
%   3) PLAY — разыграть лучшую карту из руки при текущей мане и месте на столе.
%      Эвристика карты: score = 3*A + 2*H - Cost
%      (+3 taunt, +2 charge, +2 rush — если такие способности описаны в БЗ).
%   4) END_TURN — если больше ничего сделать нельзя.
%
% Гарантии и устойчивость:
%   • Детерминированный единичный шаг (без перебора состояний).
%   • Всегда связывает NewState (если ничего не удалось — возвращает исходное State).
%   • Все вызовы движка обёрнуты в once(catch(...)) — безопасные отказ/исключения.
%   • Учитываются "странные" термы: проверки is_list/1, compound/1 и т.п.
%   • Сообщения печатаются на английском ASCII — без проблем с кодировкой в Windows.
%
% Экспортируемые предикаты (API):
%   advise_now/1     — печатает совет, не применяя действие.
%   do_best_now/2    — применяет лучшее доступное действие (гарантирует NewState).
%   step/2           — алиас к do_best_now/2.
%   auto/3           — сделать N последовательных жадных шагов.
%
% Зависимости и запуск:
%   В REPL:
%     ?- ['01/hearthstone.pl'].
%     ?- use_module('02/advisor.pl').
%     ?- initial_state(S0), advise_now(S0).
%     ?- do_best_now(S0, S1), display_game(S1).
%
% Ограничения текущей версии:
%   • Нет глубокого поиска, только локальная жадная эвристика.
% =========================================================

:- encoding(utf8).

:- module(advisor, [
    advise_now/1,        % advise_now(+State) : печатает, какое действие будет выбрано
    do_best_now/2,       % do_best_now(+State, -NewState) : применяет лучшее доступное действие
    step/2,              % step(+State, -NewState) : алиас к do_best_now/2
    auto/3               % auto(+N, +State, -OutState) : сделать N шагов подряд
]).

:- use_module(library(lists)).

% -----------------------
% ПУБЛИЧНЫЙ ИНТЕРФЕЙС
% -----------------------

% advise_now(+State)
% Печатает совет (что именно будет сделано) без применения хода.
advise_now(State) :-
    ( plan_kill(State, Att, Tgt) ->
        format('Advice: attack(~w,~w) - kill target ~w~n', [Att, Tgt, Tgt]), !
    ; plan_face(State, AttFace) ->
        format('Advice: attack(~w,0) - go face~n', [AttFace]), !
    ; plan_play(State, Card) ->
        format('Advice: play(~w) - develop board~n', [Card]), !
    ; format('Advice: end_turn - no other legal moves~n', [])
    ).

% do_best_now(+State, -NewState)
% Применяет ПЕРВОЕ легальное действие по приоритетам (kill/face/play/end_turn).
% ГАРАНТИЯ: всегда связывает NewState (если ничего не удалось — вернёт State).
do_best_now(State, NewState) :-
    ( catch(try_kill(State, S2),  _, fail) -> NewState = S2
    ; catch(try_face(State, S3),  _, fail) -> NewState = S3
    ; catch(try_play(State, S4),  _, fail) -> NewState = S4
    ; catch(end_turn(State, S5),   _, fail) -> NewState = S5
    ; NewState = State
    ).

% step/2 — удобный алиас
step(S, S1) :- do_best_now(S, S1).

% auto(+N, +S, -Out) — сделать N шагов подряд (жадно)
auto(0, S, S) :- !.
auto(N, S, Out) :-
    N > 0,
    step(S, S1),
    N1 is N - 1,
    auto(N1, S1, Out).

% -----------------------
% ПРИОРИТЕТ 1: KILL
% -----------------------

% try_kill(+State, -NewState)
% Пытается выполнить убийство выбранной цели: attack(Att, Tgt).
try_kill(State, S2) :-
    plan_kill(State, Att, Tgt),
    once(catch(attack(State, Att, Tgt, S2), _, fail)).

% plan_kill(+State, -AttBest, -TgtBest)
% Планирование убийства БЕЗ вызова attack/4:
% • ранжируем цели по (2*HP + ATK) с бонусом за taunt
% • выбираем МОЕГО ГОТОВОГО атакующего, который добивает цель (A >= HP_цели)
plan_kill(State, AttBest, TgtBest) :-
    opp_board(State, OB), is_list(OB), OB \= [],
    findall(Val-TgtIdx,
        ( opp_target_info(State, TgtIdx, ATgt, HTgt, Name),
          V0 is HTgt*2 + ATgt,
          ( ability_of(Name, taunt) -> Val is V0 + 5 ; Val = V0 )
        ),
        Pairs),
    Pairs \= [],
    keysort(Pairs, Sorted), reverse(Sorted, Desc),
    member(_-TgtBest, Desc),
    my_attackers_by_power(State, AttackersDesc),
    opp_target_info(State, TgtBest, _AT, HT, _Name),
    member(AttBest, AttackersDesc),
    my_board(State, MB), nth1(AttBest, MB, MinA),
    MinA = minion(_, _, _, CanA, _), CanA == true,         % должен быть готов
    attacker_attack(State, AttBest, A),
    A >= HT, !.

% -----------------------
% ПРИОРИТЕТ 2: FACE (если нет таунта)
% -----------------------

% try_face(+State, -NewState)
% Пытается ударить в лицо самым сильным ГОТОВЫМ атакующим (без RushOnly).
try_face(State, S2) :-
    plan_face(State, Att),
    once(catch(attack(State, Att, 0, S2), _, fail)).

% plan_face(+State, -AttBest)
% Разрешено только если у оппонента нет таунтов; берём МОЕГО ГОТОВОГО атакующего
% с максимальной атакой (>0) и без RushOnly (в движке нельзя бить лицо с RushOnly).
plan_face(State, AttBest) :-
    \+ opp_has_taunt(State),
    my_board(State, MB),
    is_list(MB), MB \= [],
    findall(A-Idx,
        ( nth1(Idx, MB, Min),
          Min = minion(_, _, _, Can, RushOnly),
          Can == true, RushOnly == false,
          minion_attack_current(Min, A),
          A > 0
        ),
        Pairs),
    Pairs \= [],
    keysort(Pairs, Sorted),
    last(Sorted, _-AttBest), !.

% -----------------------
% ПРИОРИТЕТ 3: PLAY (лучшая карта по эвристике)
% -----------------------

% try_play(+State, -NewState)
% Разыгрываем вычисленную лучшую карту: play_minion/3 вызов ровно один раз.
try_play(State, S2) :-
    plan_play(State, Card),
    once(catch(play_minion(State, Card, S2), _, fail)).

% plan_play(+State, -BestName)
% Выбор лучшей карты из руки с учётом маны и лимита стола:
% score = 3*A + 2*H - Cost (+ бонусы за taunt/charge/rush, если заданы в БЗ).
plan_play(State, BestName) :-
    my_player(State, player(_H, Mana, _Max, _Deck, Hand, Board)),
    is_list(Board), length(Board, L), L < 7,
    is_list(Hand),
    findall(Score-Name,
        ( member(CT, Hand),
          card_term_name(CT, Name),
          card_term_cost(CT, Cost),
          number(Mana), number(Cost), Cost =< Mana,
          card_attack(Name, A), card_health(Name, H),
          ( ability_of(Name, taunt)  -> Ta = 3 ; Ta = 0 ),
          ( ability_of(Name, charge) -> Ch = 2 ; Ch = 0 ),
          ( ability_of(Name, rush)   -> Ru = 2 ; Ru = 0 ),
          Score is 3*A + 2*H + Ta + Ch + Ru - Cost
        ),
        Pairs),
    Pairs \= [],
    keysort(Pairs, Sorted), last(Sorted, _-BestName), !.

% -----------------------
% ВСПОМОГАТЕЛЬНЫЕ ПРЕДИКАТЫ
% -----------------------

% Текущий и оппонент (деконструкция game_state/4)
my_player(game_state(P1, _P2, 1, _), P1).
my_player(game_state(_P1, P2, 2, _), P2).
opp_player(game_state(P1, _P2, 2, _), P1).
opp_player(game_state(_P1, P2, 1, _), P2).

% Доски текущего/оппонента
my_board(State, B)  :- my_player(State, player(_,_,_,_,_,B)).
opp_board(State, B) :- opp_player(State, player(_,_,_,_,_,B)).

% Индексы существ на досках (1..N), устойчиво к не-спискам
board_indices(State, MyIdxs, OppIdxs) :-
    ( my_board(State, MB), is_list(MB) -> length(MB, ML), numlist(1, ML, MyIdxs) ; MyIdxs = [] ),
    ( opp_board(State, OB), is_list(OB) -> length(OB, OL), numlist(1, OL, OppIdxs) ; OppIdxs = [] ).

% Мои атакующие по убыванию атаки, возвращаем список индексов
my_attackers_by_power(State, SortedIdxs) :-
    board_indices(State, MyIdxs, _),
    findall(A-Idx, (member(Idx, MyIdxs), attacker_attack(State, Idx, A)), P),
    keysort(P, S), reverse(S, D),
    findall(Idx, member(_-Idx, D), SortedIdxs).

% Информация о цели на доске оппонента по индексу
opp_target_info(State, TgtIdx, A, H, Name) :-
    board_indices(State, _MyIdxs, OppIdxs),
    member(TgtIdx, OppIdxs),
    opp_board(State, OB), nth1(TgtIdx, OB, M),
    minion_attack_current(M, A),
    minion_health_current(M, H),
    minion_name_only(M, Name).

% Проверка наличия таунта у оппонента (если в БЗ нет taunt — просто провал)
opp_has_taunt(State) :-
    opp_board(State, OB),
    is_list(OB), OB \= [],
    member(M, OB),
    minion_name_only(M, N),
    N \= unknown,
    ability_of(N, taunt), !.

% Текущие статы миньона; безопасно обрабатывает «плейсхолдеры»
attacker_attack(State, AttIdx, A) :-
    my_board(State, MB), nth1(AttIdx, MB, M),
    minion_attack_current(M, A).

% ВНИМАНИЕ: структура minion/5 — minion(Name, Attack, Health, CanAttack, RushOnly)
minion_attack_current(M, A) :-
    ( compound(M), arg(2, M, A0), number(A0) -> A = A0
    ; compound(M), arg(1, M, Name)           -> card_attack(Name, A)
    ; A = 0
    ).

minion_health_current(M, H) :-
    ( compound(M), arg(3, M, H0), number(H0) -> H = H0
    ; compound(M), arg(1, M, Name)           -> card_health(Name, H)
    ; H = 0
    ).

minion_name_only(M, Name) :-
    ( compound(M), arg(1, M, Name) -> true ; Name = unknown ).

% Извлечение имени/стоимости карты из терма руки
card_term_name(card(Name, _Cost, _A, _H), Name) :- !.
card_term_name(Name, Name).
card_term_cost(card(_Name, Cost, _A, _H), Cost) :- !.
card_term_cost(_Name, _) :- fail.

