import { Component, inject } from '@angular/core';

import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatCardModule } from '@angular/material/card';
import { MatSelectModule } from '@angular/material/select';

import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { MarkdownModule } from 'ngx-markdown';
import { ChatService } from './core/services/chat.service';
import { ChatModelStrategyEnum } from './shared/models/chat-model-strategy.enum';
import { ChatModelTypeEnum } from './shared/models/chat-model-type.enum';
import { HttpClientModule } from '@angular/common/http';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatListModule,
    MatCardModule,
    FormsModule,
    CommonModule,
    MarkdownModule,
    MatSelectModule,
    HttpClientModule
  ],
  providers: [ChatService],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  message: string = '';
  messages: { text: string; fromUser: boolean }[] = [];
  strategiesOptions = [
    {type: ChatModelStrategyEnum.VECTOR, value: "Vector"},
    {type: ChatModelStrategyEnum.VECTOR_BM25, value: "Vector + BM25"},
    {type: ChatModelStrategyEnum.VECTOR_RERANK, value: "Vector + Rerank"},
    {type: ChatModelStrategyEnum.VECTOR_BM25_RERANK, value: "Vector + BM25 + Rerank"},
  ]
  selectedStrategyOption = this.strategiesOptions[0]

  modelTypeOptions = [
    {type: ChatModelTypeEnum.GPT_4O_MINI, value: "GPT4o-mini"},
    {type: ChatModelTypeEnum.FINE_TUNED_GPT_40_MINI, value: "Fine-Tuned GPT4o-mini"}
  ]
  selectedTypeOption = this.modelTypeOptions[0]

  pythonVersionOptions = [
    "3.10","3.11","3.12","3.13"
  ]
  pythonOption = this.pythonVersionOptions[0]

  chatService = inject(ChatService)

  sendMessage() {
    console.log(this.message)
    if (!this.message.trim()) return;

    this.messages.push({ text: this.message, fromUser: true });

    this.chatService
      .getResponse(
        this.message, 
        this.pythonOption, 
        7, 
        this.selectedStrategyOption.type, 
        this.selectedTypeOption.type)
      .subscribe(response => {
        this.messages.push({ text: response.response, fromUser: false });
      }
    )

    this.message = '';
  }
}
