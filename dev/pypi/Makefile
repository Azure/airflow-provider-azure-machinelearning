.PHONY: clean install uninstall

install:
	docker compose down
	docker compose build --no-cache --pull
	docker compose create
	docker compose start

uninstall:
	docker compose down

clean:
	docker compose down
