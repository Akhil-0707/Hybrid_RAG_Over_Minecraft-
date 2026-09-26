package com.mcrag.client;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import com.mojang.brigadier.suggestion.Suggestions;
import com.mojang.brigadier.suggestion.SuggestionsBuilder;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.MutableComponent;

import java.net.URI;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

import static net.fabricmc.fabric.api.client.command.v2.ClientCommands.argument;
import static net.fabricmc.fabric.api.client.command.v2.ClientCommands.literal;

/**
 * Client-side commands (they run on this machine, so they work on any server):
 *   /doubt <question>   ask anything; the answer comes back in chat with clickable wiki sources
 *   /faq                FAQs for the biome you are standing in
 *   /faq <biome>        FAQs for another biome, e.g. /faq cherry grove
 */
public final class McragClient implements ClientModInitializer {
	/** Override with the JVM flag -Dmcrag.backend=http://host:port when the backend moves. */
	private static final BackendClient BACKEND =
			new BackendClient(System.getProperty("mcrag.backend", "http://127.0.0.1:8765"));
	private static final Component PREFIX = Component.literal("[Wiki] ").withStyle(ChatFormatting.GOLD);
	/** When the backend was last asked to load the model (0 = it has been unloaded since). */
	private static long lastWarmup;

	@Override
	public void onInitializeClient() {
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
			dispatcher.register(literal("doubt")
					.then(argument("question", StringArgumentType.greedyString())
							.suggests(McragClient::warmUp)
							.executes(McragClient::doubt)));
			dispatcher.register(literal("faq")
					.executes(ctx -> faq(ctx, null))
					.then(argument("biome", StringArgumentType.greedyString())
							.executes(ctx -> faq(ctx, StringArgumentType.getString(ctx, "biome")))));
		});
	}

	private static int doubt(CommandContext<FabricClientCommandSource> ctx) {
		String question = StringArgumentType.getString(ctx, "question");
		Minecraft mc = Minecraft.getInstance();
		if (mc.player == null || mc.level == null) {
			return 0;
		}
		BlockPos pos = mc.player.blockPosition();
		String biome = currentBiome(mc, pos);
		String dimension = mc.level.dimension().identifier().toString();
		say(Component.literal("Thinking about \"" + question + "\"...").withStyle(ChatFormatting.GRAY));
		lastWarmup = 0;  // the backend unloads the model after answering; warm up again next time
		BACKEND.doubt(question, biome, dimension, pos.getX(), pos.getY(), pos.getZ(),
						event -> onMainThread(() -> showEvent(event)))
				.exceptionally(error -> onMainThread(() -> showError(error)));
		return 1;
	}

	/**
	 * Called as the player types the question (to offer suggestions; there are none). Loading the
	 * model takes several seconds, so the backend starts it now and the load overlaps the typing.
	 */
	private static CompletableFuture<Suggestions> warmUp(CommandContext<FabricClientCommandSource> ctx,
			SuggestionsBuilder builder) {
		long now = System.currentTimeMillis();
		if (now - lastWarmup > 60_000) {
			lastWarmup = now;
			BACKEND.warmup();
		}
		return builder.buildFuture();
	}

	private static int faq(CommandContext<FabricClientCommandSource> ctx, String biomeArg) {
		Minecraft mc = Minecraft.getInstance();
		if (mc.player == null || mc.level == null) {
			return 0;
		}
		String biome = biomeArg != null ? biomeArg : currentBiome(mc, mc.player.blockPosition());
		if (biome == null) {
			say(Component.literal("Couldn't tell which biome you're in.").withStyle(ChatFormatting.RED));
			return 0;
		}
		BACKEND.faq(biome)
				.thenAccept(json -> onMainThread(() -> showFaq(json)))
				.exceptionally(error -> onMainThread(() -> showError(error)));
		return 1;
	}

	private static String currentBiome(Minecraft mc, BlockPos pos) {
		return mc.level.getBiome(pos).unwrapKey().map(key -> key.identifier().toString()).orElse(null);
	}

	/** One streamed event: a line of the answer, the closing sources, or an error. */
	private static void showEvent(JsonObject event) {
		if (event.has("line")) {
			String line = event.get("line").getAsString();
			if (!line.isBlank()) {
				say(Component.literal(line));
			}
		} else if (event.has("error")) {
			say(Component.literal(event.get("error").getAsString()).withStyle(ChatFormatting.RED));
		} else if (event.has("done")) {
			for (JsonElement element : event.getAsJsonArray("sources")) {
				JsonObject source = element.getAsJsonObject();
				String label = "[" + source.get("n").getAsInt() + "] " + source.get("title").getAsString()
						+ " > " + source.get("section").getAsString();
				say(link(label, source.get("url").getAsString()));
			}
			say(Component.literal(event.get("model").getAsString() + ", " + event.get("seconds").getAsDouble() + " s")
					.withStyle(ChatFormatting.DARK_GRAY));
		}
	}

	private static void showFaq(JsonObject json) {
		say(Component.literal("FAQs: ").withStyle(ChatFormatting.YELLOW)
				.append(link(json.get("title").getAsString(), json.get("url").getAsString())));
		for (JsonElement element : json.getAsJsonArray("faqs")) {
			JsonObject item = element.getAsJsonObject();
			say(Component.literal("Q: " + item.get("q").getAsString()).withStyle(ChatFormatting.AQUA));
			say(Component.literal("A: " + item.get("a").getAsString()));
		}
	}

	private static void showError(Throwable error) {
		Throwable cause = error instanceof CompletionException && error.getCause() != null ? error.getCause() : error;
		String message = cause instanceof BackendClient.BackendException
				? cause.getMessage()
				: "Couldn't reach the backend at " + BACKEND.baseUrl() + " - is `python -m mcrag serve` running?";
		say(Component.literal(message).withStyle(ChatFormatting.RED));
	}

	private static MutableComponent link(String label, String url) {
		return Component.literal(label).withStyle(style -> style
				.withColor(ChatFormatting.BLUE)
				.withUnderlined(true)
				.withClickEvent(new ClickEvent.OpenUrl(URI.create(url))));
	}

	/** Shows a message in this player's own chat (nothing is sent to the server). */
	private static void say(Component message) {
		Minecraft mc = Minecraft.getInstance();
		if (mc.player != null) {
			mc.player.sendSystemMessage(Component.empty().append(PREFIX).append(message));
		}
	}

	/** Chat must be touched on the game thread; HTTP callbacks arrive on other threads. */
	private static Void onMainThread(Runnable task) {
		Minecraft.getInstance().execute(task);
		return null;
	}
}
