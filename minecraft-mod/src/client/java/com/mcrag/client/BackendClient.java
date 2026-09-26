package com.mcrag.client;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

/** Talks to the local Python backend (`python -m mcrag serve`) without blocking the game thread. */
public final class BackendClient {
	private static final Gson GSON = new Gson();
	private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
	private final String baseUrl;

	public BackendClient(String baseUrl) {
		this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
	}

	public String baseUrl() {
		return baseUrl;
	}

	/** POST /doubt with the question and where the player is. */
	public CompletableFuture<JsonObject> doubt(String question, String biome, String dimension, int x, int y, int z) {
		JsonObject body = new JsonObject();
		body.addProperty("question", question);
		if (biome != null) body.addProperty("biome", biome);
		if (dimension != null) body.addProperty("dimension", dimension);
		body.addProperty("x", x);
		body.addProperty("y", y);
		body.addProperty("z", z);
		HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/doubt"))
				.timeout(Duration.ofSeconds(180))  // a cold local model can take a minute or more
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(GSON.toJson(body)))
				.build();
		return send(request);
	}

	/** GET /faq?biome=... (an in-game id such as minecraft:cherry_grove, or a biome name). */
	public CompletableFuture<JsonObject> faq(String biome) {
		String query = URLEncoder.encode(biome, StandardCharsets.UTF_8);
		HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/faq?biome=" + query))
				.timeout(Duration.ofSeconds(15))
				.GET()
				.build();
		return send(request);
	}

	private CompletableFuture<JsonObject> send(HttpRequest request) {
		return http.sendAsync(request, HttpResponse.BodyHandlers.ofString()).thenApply(response -> {
			JsonObject json;
			try {
				json = GSON.fromJson(response.body(), JsonObject.class);
			} catch (RuntimeException e) {
				json = null;
			}
			if (response.statusCode() != 200) {
				String detail = json != null && json.has("detail") ? json.get("detail").toString() : response.body();
				throw new BackendException("HTTP " + response.statusCode() + ": " + detail);
			}
			if (json == null) {
				throw new BackendException("the backend returned an unreadable response");
			}
			return json;
		});
	}

	/** An error the backend reported (as opposed to it being unreachable). */
	public static final class BackendException extends RuntimeException {
		public BackendException(String message) {
			super(message);
		}
	}
}
