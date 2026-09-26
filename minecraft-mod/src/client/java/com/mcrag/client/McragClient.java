package com.mcrag.client;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
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
import java.util.concurrent.CompletionException;

import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.argument;
import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.literal;

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

	@Override
	public void onInitializeClient() {
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
			dispatcher.register(literal("doubt")
					.then(argument("question", StringArgumentType.greedyString())
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
		BACKEND.doubt(question, biome, dimension, pos.getX(), pos.getY(), pos.getZ())
				.thenAccept(json -> onMainThread(() -> showAnswer(json)))
				.exceptionally(error -> onMainThread(() -> showError(error)));
		return 1;
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

	private static void showAnswer(JsonObject json) {
		for (String line : json.get("answer").getAsString().split("\n")) {
			if (!line.isBlank()) {
				say(Component.literal(line));
			}
		}
		for (JsonElement element : json.getAsJsonArray("sources")) {
			JsonObject source = element.getAsJsonObject();
			String label = "[" + source.get("n").getAsInt() + "] " + source.get("title").getAsString()
					+ " > " + source.get("section").getAsString();
			say(link(label, source.get("url").getAsString()));
		}
		say(Component.literal(json.get("model").getAsString() + ", " + json.get("seconds").getAsDouble() + " s")
				.withStyle(ChatFormatting.DARK_GRAY));
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

	private static void say(Component message) {
		Minecraft.getInstance().gui.getChat().addMessage(Component.empty().append(PREFIX).append(message));
	}

	/** Chat must be touched on the game thread; HTTP callbacks arrive on other threads. */
	private static Void onMainThread(Runnable task) {
		Minecraft.getInstance().execute(task);
		return null;
	}
}
