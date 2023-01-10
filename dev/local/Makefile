.PHONY: wheel clean install uninstall

install: wheel
	docker compose down
	docker compose build --no-cache --pull
	docker compose create
	docker compose start

uninstall:
	docker compose down

wheel: clean
	make -C ../..
	cp ../../dist/airflow_provider_azure_machinelearning*.whl .

clean:
	rm -f ./airflow_provider_azure_machinelearning*.whl
	make clean -C ../..
