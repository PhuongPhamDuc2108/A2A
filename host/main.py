import gradio as gr
import httpx
import asyncio
import os
import json
from dotenv import load_dotenv

# Tải các biến môi trường và import HostAgent
# Đảm bảo tệp này được đặt trong cùng thư mục gốc với thư mục 'app'
load_dotenv()
from agent import HostAgent


# Lớp quản lý trạng thái của các agent trong UI
class AgentManager:
    """Quản lý danh sách các remote agent được thêm vào từ UI."""

    def __init__(self):
        self.remote_agent_urls = set()

    async def fetch_agent_card(self, base_url: str):
        """Lấy thông tin AgentCard từ endpoint .well-known."""
        well_known_url = f"{base_url.rstrip('/')}/.well-known/a2a/agent-card"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(well_known_url)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            return {"error": f"Không thể kết nối đến agent: {e}"}
        except Exception as e:
            return {"error": f"Lỗi không xác định: {e}"}

    def add_agent(self, url: str):
        """Thêm một URL agent mới vào danh sách."""
        if url:
            self.remote_agent_urls.add(url.strip())
        return list(self.remote_agent_urls)

    def get_agents_as_str(self):
        """Trả về danh sách URL agent dưới dạng chuỗi cho HostAgent."""
        return ",".join(self.remote_agent_urls)


# Khởi tạo đối tượng quản lý
agent_manager = AgentManager()


# --- Các hàm xử lý sự kiện cho Gradio ---

async def add_agent_ui(url: str):
    """
    Hàm được gọi khi người dùng nhấn nút 'Thêm Agent'.
    Nó sẽ lấy AgentCard và cập nhật danh sách agent.
    """
    if not url:
        return agent_manager.remote_agent_urls, gr.update(value="URL không được để trống.")

    card_data = await agent_manager.fetch_agent_card(url)
    if "error" in card_data:
        return list(agent_manager.remote_agent_urls), json.dumps(card_data, indent=2)

    updated_list = agent_manager.add_agent(url)
    return updated_list, json.dumps(card_data, indent=2)


async def chat_function(message: str, history: list):
    """
    Hàm chính xử lý việc chat.
    Nó sẽ khởi tạo HostAgent với danh sách agent hiện tại và stream phản hồi.
    """
    if not agent_manager.get_agents_as_str():
        yield "Vui lòng thêm ít nhất một remote agent trong tab 'Quản lý Agents' trước khi bắt đầu chat."
        return

    # Cập nhật biến môi trường tạm thời cho HostAgent sử dụng
    os.environ["REMOTE_AGENT_URLS"] = agent_manager.get_agents_as_str()

    try:
        # Khởi tạo HostAgent với danh sách agent mới nhất
        host_agent = HostAgent()

        history.append([message, ""])
        full_response = ""

        # Stream phản hồi từ HostAgent
        async for chunk in host_agent.stream(message, "gradio-session"):
            # Nối các phần phản hồi vào chuỗi đầy đủ
            content_part = chunk.get('content', '')
            full_response += content_part + "\n"
            # Cập nhật chatbot UI với phản hồi đang được stream
            history[-1][1] = full_response
            yield history

    except Exception as e:
        history.append([message, f"Đã xảy ra lỗi: {e}"])
        yield history


# --- Xây dựng giao diện Gradio ---

with gr.Blocks(theme=gr.themes.Soft(), title="Host Agent UI") as demo:
    gr.Markdown("# Giao diện tương tác với Host Agent")

    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot(label="Conversation", height=600)
            msg_textbox = gr.Textbox(placeholder="Hỏi Host Agent điều gì đó...", label="Your Message", lines=2)
            msg_textbox.submit(chat_function, [msg_textbox, chatbot], chatbot)
            msg_textbox.submit(lambda: "", None, msg_textbox)

        with gr.TabItem("Quản lý Agents"):
            gr.Markdown("## Thêm và xem các Remote Agent")
            with gr.Row():
                agent_url_input = gr.Textbox(label="URL của Remote Agent", placeholder="http://localhost:10001")
                add_agent_btn = gr.Button("Thêm Agent")

            gr.Markdown("### Danh sách Agent đã thêm")
            agent_list_display = gr.JSON(label="Agent URLs")

            gr.Markdown("### Thông tin Agent Card (JSON)")
            agent_card_json = gr.JSON(label="Agent Card Details")

            add_agent_btn.click(
                add_agent_ui,
                inputs=[agent_url_input],
                outputs=[agent_list_display, agent_card_json]
            )

if __name__ == "__main__":
    # Chạy giao diện web Gradio
    demo.launch(server_name="0.0.0.0", server_port=7860)
