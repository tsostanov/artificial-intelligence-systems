-----------------------
ПРИМЕРЫ ЗАПРОСОВ
-----------------------
?- consult('lab1_hearthstone_charge_rush.pl').
?- initial_state(S), display_game(S).
?- initial_state(S), play_minion(S, stonetusk_boar, S2), display_game(S2).
?- initial_state(S), play_minion(S, stonetusk_boar, S2), attack(S2, 1, 0, S3), display_game(S3).
(последний пример: stonetusk_boar имеет ability charge -> CanAttack = true -> можно атаковать героя сразу)

Для rush:
добавь в руку карту с ability rush (например murloc_raider), сыграй и попытайся атаковать героя - должна быть ошибка
?- initial_state(S), play_minion(S, murloc_raider, S2), attack(S2, 1, 0, S3). % этот вызов должен fail, потому что rush не может атаковать героя
?- initial_state(S), play_minion(S, murloc_raider, S2), attack(S2, 1, 1, S3). % если у врага есть minion на позиции 1, допустимо
