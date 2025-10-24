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
%   0) PLAY-HASTE — разыгрываем карту с charge/rush, если можем (разложим стол заранее).
%   1) PLAY       — любая лучшая карта по эвристике (пока есть мана и место).
%   2) KILL       — выгодный размен (A атакера >= HP цели), атакер должен быть готов.
%   3) FACE       — удар в лицо; атакер готов и НЕ RushOnly.
%   4) END_TURN   — если больше ничего сделать нельзя.
% =========================================================

:- encoding(utf8).

:- module(advisor, [
    advise_now/1,
    do_best_now/2,
    step/2,
    auto/3
]).

:- use_module(library(lists)).

% -----------------------
% ПУБЛИЧНЫЙ ИНТЕРФЕЙС
% -----------------------

advise_now(State) :-
    ( plan_play_haste(State, CardH) ->
        format('Advice: play(~w) - develop board (haste)~n', [CardH]), !
    ; plan_play(State, Card) ->
        format('Advice: play(~w) - develop board~n', [Card]), !
    ; plan_kill(State, Att, Tgt) ->
        format('Advice: attack(~w,~w) - kill target ~w~n', [Att, Tgt, Tgt]), !
    ; plan_face(State, AttFace) ->
        format('Advice: attack(~w,0) - go face~n', [AttFace]), !
    ; format('Advice: end_turn - no other legal moves~n', [])
    ).

do_best_now(State, NewState) :-
    ( catch(try_play_haste(State, S0), _, fail) -> NewState = S0
    ; catch(try_play(State, S1),       _, fail) -> NewState = S1
    ; catch(try_kill(State, S2),       _, fail) -> NewState = S2
    ; catch(try_face(State, S3),       _, fail) -> NewState = S3
    ; catch(end_turn(State, S4),       _, fail) -> NewState = S4
    ; NewState = State
    ).

step(S, S1) :- do_best_now(S, S1).

auto(0, S, S) :- !.
auto(N, S, Out) :-
    N > 0,
    step(S, S1),
    N1 is N - 1,
    auto(N1, S1, Out).

% -----------------------
% PLAY (HASTE сначала)
% -----------------------

try_play_haste(State, S2) :-
    plan_play_haste(State, Card),
    once(catch(play_minion(State, Card, S2), _, fail)).

% Выбрать карту с charge или rush (если хватает маны и есть место).
% Приоритет: charge выше rush, затем A/H, ниже стоимость.
plan_play_haste(State, BestName) :-
    my_player(State, player(_H, Mana, _Max, _Deck, Hand, Board)),
    is_list(Board), length(Board, L), L < 7,
    is_list(Hand),
    findall(Score-Name,
        ( member(CT, Hand),
          card_term_name(CT, Name),
          card_term_cost(CT, Cost),
          number(Mana), number(Cost), Cost =< Mana,
          ( ability_of(Name, charge) -> HT = 2
          ; ability_of(Name, rush)   -> HT = 1
          ; fail
          ),
          card_attack(Name, A), card_health(Name, H),
          Score is HT*100 + 3*A + 2*H - Cost
        ),
        Pairs),
    Pairs \= [],
    keysort(Pairs, Sorted), last(Sorted, _-BestName), !.

% -----------------------
% PLAY (общая эвристика)
% -----------------------

try_play(State, S2) :-
    plan_play(State, Card),
    once(catch(play_minion(State, Card, S2), _, fail)).

% score = 3*A + 2*H - Cost (+bonuses за charge/rush)
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
          ( ability_of(Name, charge) -> Ch = 2 ; Ch = 0 ),
          ( ability_of(Name, rush)   -> Ru = 2 ; Ru = 0 ),
          Score is 3*A + 2*H + Ch + Ru - Cost
        ),
        Pairs),
    Pairs \= [],
    keysort(Pairs, Sorted), last(Sorted, _-BestName), !.

% -----------------------
% KILL
% -----------------------

try_kill(State, S2) :-
    plan_kill(State, Att, Tgt),
    once(catch(attack(State, Att, Tgt, S2), _, fail)).

% Цель — по (2*HP + ATK); атакёр — готовый и A >= HP цели.
plan_kill(State, AttBest, TgtBest) :-
    opp_board(State, OB), is_list(OB), OB \= [],
    findall(Val-TgtIdx,
        ( opp_target_info(State, TgtIdx, ATgt, HTgt, _Name),
          Val is HTgt*2 + ATgt
        ),
        Pairs),
    Pairs \= [],
    keysort(Pairs, Sorted), reverse(Sorted, Desc),
    member(_-TgtBest, Desc),
    my_attackers_by_power(State, AttackersDesc),
    member(AttBest, AttackersDesc),
    attacker_ready(State, AttBest),
    target_health_number(State, TgtBest, HT),
    attacker_attack(State, AttBest, A),
    number(A), number(HT),
    A >= HT, !.

% -----------------------
% FACE
% -----------------------

try_face(State, S2) :-
    plan_face(State, Att),
    once(catch(attack(State, Att, 0, S2), _, fail)).

% Берём САМЫЙ СИЛЬНЫЙ из «готов и не RushOnly»
plan_face(State, AttBest) :-
    ready_nonrush_attackers(State, Pairs),
    Pairs \= [],
    keysort(Pairs, Sorted),
    last(Sorted, _-AttBest), !.

% собрать всех атакующих, которые могут бить лицо прямо сейчас
ready_nonrush_attackers(State, Pairs) :-
    my_board(State, MB),
    is_list(MB), MB \= [],
    findall(A-Idx,
        ( nth1(Idx, MB, Min),
          compound(Min),
          arg(4, Min, true),   % CanAttack
          arg(5, Min, RO), RO == false,
          minion_attack_current(Min, A),
          number(A), A > 0
        ),
        Pairs).

% -----------------------
% ВСПОМОГАТЕЛЬНЫЕ ПРЕДИКАТЫ
% -----------------------

% текущий и оппонент
my_player(game_state(P1, _P2, 1, _), P1).
my_player(game_state(_P1, P2, 2, _), P2).
opp_player(game_state(P1, _P2, 2, _), P1).
opp_player(game_state(_P1, P2, 1, _), P2).

% доски
my_board(State, B)  :- my_player(State, player(_,_,_,_,_,B)).
opp_board(State, B) :- opp_player(State, player(_,_,_,_,_,B)).

% индексы существ (устойчиво)
board_indices(State, MyIdxs, OppIdxs) :-
    ( my_board(State, MB), is_list(MB) -> length(MB, ML), numlist(1, ML, MyIdxs) ; MyIdxs = [] ),
    ( opp_board(State, OB), is_list(OB) -> length(OB, OL), numlist(1, OL, OppIdxs) ; OppIdxs = [] ).

% список моих атакеров по убыванию атаки (индексы)
my_attackers_by_power(State, SortedIdxs) :-
    board_indices(State, MyIdxs, _),
    findall(A-Idx, (member(Idx, MyIdxs), attacker_attack(State, Idx, A)), P),
    keysort(P, S), reverse(S, D),
    findall(Idx, member(_-Idx, D), SortedIdxs).

% инфо о цели противника
opp_target_info(State, TgtIdx, A, H, Name) :-
    board_indices(State, _MyIdxs, OppIdxs),
    member(TgtIdx, OppIdxs),
    opp_board(State, OB), nth1(TgtIdx, OB, M),
    minion_attack_current(M, A),
    minion_health_current(M, H),
    minion_name_only(M, Name).

% атакер готов? (CanAttack==true)
attacker_ready(State, Idx) :-
    my_board(State, MB), nth1(Idx, MB, M),
    compound(M), arg(4, M, true).

% здоровье цели как ЧИСЛО
target_health_number(State, TgtIdx, HT) :-
    opp_board(State, OB), nth1(TgtIdx, OB, M),
    ( compound(M), arg(3, M, H0), number(H0) -> HT = H0
    ; compound(M), arg(1, M, Name)           -> card_health(Name, HT)
    ; HT = 0 ).

% текущие статы миньона
% ВНИМАНИЕ: структура minion/5 — minion(Name, Attack, Health, CanAttack, RushOnly)
attacker_attack(State, AttIdx, A) :-
    my_board(State, MB), nth1(AttIdx, MB, M),
    minion_attack_current(M, A).

minion_attack_current(M, A) :-
    ( compound(M), arg(2, M, A0), number(A0) -> A = A0
    ; compound(M), arg(1, M, Name)           -> card_attack(Name, A)
    ; A = 0 ).

minion_health_current(M, H) :-
    ( compound(M), arg(3, M, H0), number(H0) -> H = H0
    ; compound(M), arg(1, M, Name)           -> card_health(Name, H)
    ; H = 0 ).

minion_name_only(M, Name) :-
    ( compound(M), arg(1, M, Name) -> true ; Name = unknown ).

% имя/стоимость карты из терма руки
card_term_name(card(Name, _Cost, _A, _H), Name) :- !.
card_term_name(Name, Name).
card_term_cost(card(_Name, Cost, _A, _H), Cost) :- !.
card_term_cost(_Name, _) :- fail.
